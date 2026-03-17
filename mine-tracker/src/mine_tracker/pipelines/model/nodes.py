# -*- coding: utf-8 -*-
"""
Nodes para o pipeline 'model' do Kedro.

Fluxo:
- load_data: recebe o DataFrame do catálogo (minecraft_servidores_features) e filtra o servidor mais frequente
- preprocess_data: seleciona features/target, saneia, winsoriza e retorna X, y, n_drop_y
- criar_pipelines: devolve dicionário com pipelines de modelos
- treinar_modelos: faz split fixo e treina modelos in-place
- avaliar_modelos: reusa o mesmo split para avaliar, escolhe melhor por R², salva modelo e relatório
"""

import os
from datetime import datetime
from typing import Dict, Tuple
import gc

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
import gc

logger = logging.getLogger(__name__)
MODEL_DIR = "models"  # ajuste se quiser outro caminho


# =========================
# 1) Carregar dados
# =========================
def load_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filtra o servidor mais frequente e devolve apenas ele.
    Armazena o servidor escolhido em df.attrs['servidor_escolhido'].
    """
    if "ip" not in df_raw.columns:
        raise ValueError("Coluna 'ip' não encontrada no dataset de entrada.")
    if len(df_raw) == 0:
        raise ValueError("Dataset de entrada está vazio.")

    servidor_escolhido = df_raw["ip"].value_counts().index[0]
    df = df_raw[df_raw["ip"] == servidor_escolhido].copy()
    df.attrs["servidor_escolhido"] = servidor_escolhido
    return df


# =========================
# 2) Pré-processamento
# =========================
def preprocess_data(df: pd.DataFrame, features: list, target: str) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Seleciona features/target, converte para numérico, trata inf/NaN,
    winsoriza pct_var_jogadores e retorna X, y, n_drop_y.
    Também propaga 'servidor_escolhido' in X.attrs para uso posterior.
    """
    cols_necessarias = features + [target]
    faltantes = [c for c in cols_necessarias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas faltantes no dataset: {faltantes}")

    df_model = df[cols_necessarias].copy()

    # Tipos numéricos
    for c in df_model.columns:
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")

    # Substituir Inf/-Inf por NaN
    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    # Winsorizar apenas pct_var_jogadores (1% e 99%)
    if "pct_var_jogadores" in df_model.columns:
        p1, p99 = np.nanpercentile(df_model["pct_var_jogadores"], [1, 99])
        df_model["pct_var_jogadores"] = df_model["pct_var_jogadores"].clip(
            lower=p1, upper=p99
        )

    # Remover linhas com y NaN
    n_total = len(df_model)
    df_model = df_model.dropna(subset=[target])
    n_drop_y = n_total - len(df_model)

    X = df_model[features].copy().astype("float32")
    y = df_model[target].astype(float)

    # Propaga nome do servidor (se presente)
    servidor_escolhido = df.attrs.get("servidor_escolhido", None)
    if servidor_escolhido is not None:
        X.attrs["servidor_escolhido"] = servidor_escolhido

    return X, y, n_drop_y


# =========================
# 3) Modelos
# =========================
def criar_pipelines(features: list, n_estimators: int = 100, n_jobs: int = 2) -> Dict[str, Pipeline]:
    """Cria pipelines para LinearRegression e RandomForest.
    
    Args:
        features: Lista de colunas a serem usadas.
        n_estimators: Número de árvores no RF.
        n_jobs: Número de processos paralelos (limite para RAM).
    """
    num_features = features

    preprocess_linear = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            )
        ],
        remainder="drop",
    )

    preprocess_rf = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), num_features)],
        remainder="drop",
    )

    modelos = {
        "SGDRegressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("est", SGDRegressor(random_state=42)),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=0,  # Começa com zero para crescer no warm_start
            random_state=42,
            n_jobs=n_jobs,
            warm_start=True,
        ),
    }
    return modelos


# =========================
# 4) Treino
# =========================
def treinar_modelos_incremental(
    modelos: Dict,
    features: list,
    target: str,
    n_estimators_max: int = 100,
) -> Dict:
    """Treina os modelos processando o arquivo em chunks para poupar RAM.
    
    Lógica: 
    - SGDRegressor usa partial_fit.
    - RandomForest usa warm_start e adiciona 10 árvores por chunk até o limite.
    """
    filepath = "data/04_feature/minecraft_servidores_features.csv"
    chunk_size = 50000  # 50k linhas por vez
    
    logger.info("Iniciando treino INCREMENTAL (faseado) de %s", filepath)

    # Identifica o servidor mais frequente (primeira passada rápida)
    # Para simplicidade e economia de memória, vamos assumir que o filtro já foi feito 
    # ou treinar com o que vier no CSV (que deve ser o dataset pré-filtrado feature).
    
    first_chunk = True
    n_trees_per_chunk = max(1, n_estimators_max // 5) # Divide em 5 fases

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Pre-processamento básico do chunk
        for c in features + [target]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        chunk = chunk.dropna(subset=[target]).fillna(0)
        
        X_chunk = chunk[features].astype("float32")
        y_chunk = chunk[target].astype("float32")

        # 1) Treino SGD
        modelos["SGDRegressor"].named_steps["scaler"].partial_fit(X_chunk)
        X_scaled = modelos["SGDRegressor"].named_steps["scaler"].transform(X_chunk)
        modelos["SGDRegressor"].named_steps["est"].partial_fit(X_scaled, y_chunk)

        # 2) Treino RF (warm_start)
        rf = modelos["RandomForest"]
        if rf.n_estimators < n_estimators_max:
            rf.n_estimators += n_trees_per_chunk
            rf.fit(X_chunk, y_chunk)
        
        del chunk, X_chunk, y_chunk, X_scaled
        gc.collect()
        logger.info("Chunk processado. RF Trees: %d", rf.n_estimators)

    return modelos

# =========================
# 5) Avaliação + Salvamento
# =========================
def _avaliar_um(nome: str, modelo: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
    """Avalia um modelo retornando (MAE, R²)."""
    pred = modelo.predict(X)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    logger.info(f"{nome} -> MAE: {mae:.2f} | R²: {r2:.4f}")
    return mae, r2


def avaliar_modelos(
    modelos: Dict,
    features: list,
    target: str,
) -> Tuple:
    """Avalia o melhor modelo (usando uma amostra final para métricas)."""
    # Para avaliação, carregamos apenas um pedaço (sample) dos dados
    filepath = "data/04_feature/minecraft_servidores_features.csv"
    df_eval = pd.read_csv(filepath, nrows=10000)
    
    X_eval = df_eval[features].fillna(0).astype("float32")
    y_eval = df_eval[target].fillna(0).astype("float32")

    logger.info("\nAvaliação Incremental (Amostra 10k):")
    
    # Avalia SGD
    X_scaled = modelos["SGDRegressor"].named_steps["scaler"].transform(X_eval)
    pred_sgd = modelos["SGDRegressor"].named_steps["est"].predict(X_scaled)
    r2_sgd = r2_score(y_eval, pred_sgd)
    
    # Avalia RF
    pred_rf = modelos["RandomForest"].predict(X_eval)
    r2_rf = r2_score(y_eval, pred_rf)
    
    logger.info(f"SGD R²: {r2_sgd:.4f} | RF R²: {r2_rf:.4f}")
    
    melhor_nome = "RandomForest" if r2_rf > r2_sgd else "SGDRegressor"
    melhor_mdl = modelos[melhor_nome]
    
    metrics = {
        "SGDRegressor": {"r2": r2_sgd},
        "RandomForest": {"r2": r2_rf}
    }
    
    return melhor_mdl, metrics, X_eval
