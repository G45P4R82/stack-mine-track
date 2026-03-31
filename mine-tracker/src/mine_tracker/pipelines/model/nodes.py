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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, TimeSeriesSplit
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import gc
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# MLflow config - user specified http://localhost:5000
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("minecraft_mine_tracker")

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

    # Remover linhas com y NaN ou 0 (servidor offline falso positivo)
    n_total = len(df_model)
    df_model = df_model.dropna(subset=[target])
    df_model = df_model[df_model[target] > 0]
    n_drop_y = n_total - len(df_model)

    X = df_model[features].fillna(0).astype("float32")
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
                ("est", SGDRegressor(random_state=42, learning_rate='adaptive', eta0=0.001, max_iter=2000)),
            ]
        ),
        "PassiveAggressive": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("est", PassiveAggressiveRegressor(random_state=42, max_iter=2000)),
            ]
        ),
        "MLPRegressor": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                # Uma rede neural pequena, com early_stopping para evitar warnings de convergência
                ("est", MLPRegressor(hidden_layer_sizes=(32,), random_state=42, max_iter=1000)),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=0,  # Começa com zero para crescer no warm_start
            random_state=42,
            n_jobs=n_jobs,
            warm_start=True,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=0,
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

    for df_chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        X_chunk, y_chunk, _ = preprocess_data(df_chunk, features, target)
        
        if len(X_chunk) == 0:
            continue
            
        # Garante que não há valores absurdamente grandes para o float32 (Scikit-learn Overflow)
        # Clipamos em um range seguro mas amplo
        X_chunk = np.clip(X_chunk, -1e30, 1e30)
        y_chunk = np.clip(y_chunk, -1e30, 1e30)

        for nome, modelo in modelos.items():
            if hasattr(modelo, "named_steps") and hasattr(modelo.named_steps.get("est", None), "partial_fit"):
                # Pipelines iterativos (SGDRegressor, PassiveAggressive, MLPRegressor)
                modelo.named_steps["scaler"].partial_fit(X_chunk)
                X_scaled = modelo.named_steps["scaler"].transform(X_chunk)
                modelo.named_steps["est"].partial_fit(X_scaled, y_chunk)
            elif hasattr(modelo, "warm_start") and getattr(modelo, "warm_start", False):
                # Arvores baseadas em warm_start (RandomForest, ExtraTrees)
                if getattr(modelo, "n_estimators", n_estimators_max) < n_estimators_max:
                    modelo.n_estimators += n_trees_per_chunk
                    modelo.fit(X_chunk, y_chunk)
            elif hasattr(modelo, "partial_fit"):
                modelo.partial_fit(X_chunk, y_chunk)
        
        del df_chunk, X_chunk, y_chunk
        gc.collect()

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
    # Para avaliação, carregamos mais dados para ter tempo suficiente de análise
    filepath = "data/04_feature/minecraft_servidores_features.csv"
    df_eval = pd.read_csv(filepath, nrows=20000, low_memory=False)
    
    if 'timestamp' in df_eval.columns:
        df_eval['timestamp'] = pd.to_datetime(df_eval['timestamp'], errors='coerce')
        df_eval = df_eval.dropna(subset=['timestamp']).sort_values('timestamp')
    
    df_eval = df_eval.replace([np.inf, -np.inf], np.nan)
    df_eval = df_eval.dropna(subset=[target])
    df_eval = df_eval[df_eval[target] > 0]
    X_eval = df_eval[features].fillna(0).astype("float32")
    y_eval = df_eval[target].astype("float32")

    with mlflow.start_run(run_name="model_evaluation_run"):
        # Loga parâmetros
        mlflow.log_params({"features": features, "target": target, "eval_samples": len(df_eval)})
        
        logger.info("\n================ Comparativo de Modelos (Validação Cruzada Temporal) ================")
        # Como os dados tem médias móveis e dependência temporal, usamos TimeSeriesSplit para evitar vazamento do futuro para o passado
        kf = TimeSeriesSplit(n_splits=5)
    
    cv_resultados = {}
    scoring_metrics = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_rmse': 'neg_root_mean_squared_error'
    }
    
    for nome, modelo_treinado in modelos.items():
        base_model = clone(modelo_treinado)
        if hasattr(base_model, "n_estimators"):
            # Para avaliação estrutural justa, usamos um n_estimators representativo
            base_model.n_estimators = max(100, getattr(modelo_treinado, "n_estimators", 100))
            # Desliga warm_start para o CV
            base_model.warm_start = False
            
        try:
            scores = cross_validate(base_model, X_eval, y_eval, cv=kf, scoring=scoring_metrics, n_jobs=-1, return_train_score=True)
            mean_r2 = scores['test_r2'].mean()
            mean_mae = -scores['test_neg_mae'].mean()
            mean_rmse = -scores['test_neg_rmse'].mean()
            std_r2 = scores['test_r2'].std()
            
            # Extract train scores for plotting
            train_r2 = scores['train_r2'].mean()
        except Exception as e:
            logger.warning(f"Erro no K-Fold do {nome}: {e}")
            mean_r2, std_r2, mean_mae, mean_rmse, train_r2 = -99.0, 0.0, 999.0, 999.0, -99.0
            
        cv_resultados[nome] = {
            "cv_r2_mean": float(mean_r2), "cv_r2_std": float(std_r2),
            "cv_mae_mean": float(mean_mae), "cv_rmse_mean": float(mean_rmse),
            "train_r2_mean": float(train_r2)
        }
        
        # Log metrics to MLflow nested? No, for simplicity let's log main results
        mlflow.log_metrics({
            f"{nome}_cv_r2": float(mean_r2),
            f"{nome}_cv_mae": float(mean_mae),
            f"{nome}_train_r2": float(train_r2)
        })
        
        logger.info(f"Model {nome} -> K-Fold R²: {mean_r2:.4f} (Train R²: {train_r2:.4f}) | MAE: {mean_mae:.4f} | RMSE: {mean_rmse:.4f}")
    logger.info("============================================================\n")

    # ========= PLOT: Overfitting (Train vs Test R2) =========
    os.makedirs("data/08_reporting", exist_ok=True)
    
    nomes_modelos = list(cv_resultados.keys())
    test_scores = [cv_resultados[m]["cv_r2_mean"] for m in nomes_modelos]
    train_scores = [cv_resultados[m].get("train_r2_mean", 0) for m in nomes_modelos]
    
    x = np.arange(len(nomes_modelos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_scores, width, label='Train R²')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Validation R²')
    
    ax.set_ylabel('R² Score')
    ax.set_title('Comparação Cross-Validation: Train vs Validation (Overfitting Check)')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes_modelos, rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig("data/08_reporting/cv_train_vs_val_r2.png")
    plt.close()

    logger.info("Avaliação Final (Amostra 10k) - Modelos já treinados:")
    
    melhor_nome = None
    melhor_r2 = -float('inf')
    metrics = {}
    
    for nome, modelo in modelos.items():
        if hasattr(modelo, "named_steps") and "scaler" in modelo.named_steps:
            X_sc = modelo.named_steps["scaler"].transform(X_eval)
            pred = modelo.named_steps["est"].predict(X_sc)
        else:
            pred = modelo.predict(X_eval)
            
        r2_model = r2_score(y_eval, pred)
        logger.info(f"{nome} Incremental R²: {r2_model:.4f}")
        
        metrics[nome] = {
            "r2_incremental": float(r2_model),
            **cv_resultados.get(nome, {})
        }
        
        if r2_model > melhor_r2:
            melhor_r2 = r2_model
            melhor_nome = nome
            
    melhor_mdl = modelos[melhor_nome]
    
    # ========= PLOT: True vs Predicted for Best Model =========
    if hasattr(melhor_mdl, "named_steps") and "scaler" in melhor_mdl.named_steps:
        X_sc_best = melhor_mdl.named_steps["scaler"].transform(X_eval)
        pred_best = melhor_mdl.named_steps["est"].predict(X_sc_best)
    else:
        pred_best = melhor_mdl.predict(X_eval)
        
    plt.figure(figsize=(8, 8))
    plt.scatter(y_eval, pred_best, alpha=0.3, color='blue')
    plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--')
    plt.xlabel('Verdadeiro (True)')
    plt.ylabel('Previsto (Predicted)')
    plt.title(f'True vs Predicted - {melhor_nome} (Amostra 10k)')
    plt.tight_layout()
    plot_path = f"data/08_reporting/true_vs_pred_{melhor_nome}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Loga artefatos e o melhor modelo no MLflow
    try:
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact("data/08_reporting/cv_train_vs_val_r2.png")
        
        signature = infer_signature(X_eval, pred_best)
        mlflow.sklearn.log_model(melhor_mdl, "best_model", signature=signature)
        
        # Finaliza o run logando o melhor R2 global
        mlflow.log_metric("best_r2_final", float(melhor_r2))
        mlflow.set_tag("best_algorithm", melhor_nome)
    except Exception as e:
        logger.warning(f"MLflow artifact/model logging falhou (provavelmente permissão): {e}")
        # Ainda loga as métricas escalares que não dependem de disco
        try:
            mlflow.log_metric("best_r2_final", float(melhor_r2))
            mlflow.set_tag("best_algorithm", melhor_nome)
        except Exception:
            pass
    
    return melhor_mdl, metrics, X_eval
