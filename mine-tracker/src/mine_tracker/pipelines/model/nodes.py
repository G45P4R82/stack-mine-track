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

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)
# Configs básicas
FEATURES = [
    "hora",
    "final_de_semana",
    "media_movel_10",
    "proporcao_rede",
    "pct_var_jogadores",
]
TARGET = "playerCount"
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
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, int]:
    """Seleciona features/target, converte para numérico, trata inf/NaN,
    winsoriza pct_var_jogadores e retorna X, y, n_drop_y.
    Também propaga 'servidor_escolhido' em X.attrs para uso posterior.
    """
    cols_necessarias = FEATURES + [TARGET]
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
    df_model = df_model.dropna(subset=[TARGET])
    n_drop_y = n_total - len(df_model)

    X = df_model[FEATURES].copy()
    y = df_model[TARGET].astype(float)

    # Propaga nome do servidor (se presente)
    servidor_escolhido = df.attrs.get("servidor_escolhido", None)
    if servidor_escolhido is not None:
        X.attrs["servidor_escolhido"] = servidor_escolhido

    return X, y, n_drop_y


# =========================
# 3) Modelos
# =========================
def criar_pipelines() -> Dict[str, Pipeline]:
    """Cria pipelines para LinearRegression e RandomForest."""
    num_features = FEATURES

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
        "LinearRegression": Pipeline(
            steps=[
                ("prep", preprocess_linear),
                ("est", LinearRegression()),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocess_rf),
                (
                    "est",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return modelos


# =========================
# 4) Treino
# =========================
def treinar_modelos(modelos: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> Dict[str, Pipeline]:
    """Treina todos os modelos e retorna o dicionário treinado."""
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    for _, modelo in modelos.items():
        modelo.fit(X_train, y_train)
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
    modelos: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_drop_y: int,
) -> None:
    """Usa o mesmo split para avaliação, escolhe melhor por R² e salva artefatos."""
    # Mesmo split do treino
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("\nAvaliação (teste):")
    metricas = {nome: _avaliar_um(nome, mdl, X_test, y_test) for nome, mdl in modelos.items()}

    # Escolhe melhor por R²
    melhor = max(metricas, key=lambda k: metricas[k][1])

    # Salva
    os.makedirs(MODEL_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_model = os.path.join(MODEL_DIR, f"{melhor}_{ts}.joblib")
    joblib.dump(modelos[melhor], path_model)

    servidor_escolhido = X.attrs.get("servidor_escolhido", "<desconhecido>")
    path_log = os.path.join(MODEL_DIR, f"report_{ts}.txt")
    with open(path_log, "w", encoding="utf-8") as f:
        f.write("TREINO REGRESSÃO – Previsão de jogadores por horário\n")
        f.write(f"Servidor: {servidor_escolhido}\n")
        f.write(f"Linhas removidas por y NaN: {n_drop_y}\n")
        f.write("\nMétricas (teste):\n")
        for k, (mae, r2) in metricas.items():
            f.write(f"{k}: MAE={mae:.2f} | R2={r2:.4f}\n")
        f.write(f"\nMelhor modelo: {melhor}\nSalvo em: {path_model}\n")

    logger.info(f"\nMelhor modelo: {melhor}")
    logger.info(f"Modelo salvo em: {path_model}")
    logger.info(f"Relatório salvo em: {path_log}")

    # Exemplos de previsão (5 primeiros do teste)
    logger.info("\nExemplos de previsão (primeiros 5 do teste):")
    n = min(5, len(X_test))
    pred = modelos[melhor].predict(X_test.iloc[:n])
    for i, (real, prev) in enumerate(zip(y_test.iloc[:n].values, pred), start=1):
        logger.info(f"{i:02d}) Real={real:.0f} | Previsto={prev:.0f}")
