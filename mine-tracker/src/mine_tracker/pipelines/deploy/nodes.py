# -*- coding: utf-8 -*-
"""Deploy no MLflow lendo latest_model_path do parameters."""

import logging
from pathlib import Path
from typing import Dict

import joblib
import mlflow
from urllib.parse import urlparse
import socket

logger = logging.getLogger(__name__)

def _validate_tracking_uri(uri: str) -> None:
    u = urlparse(uri)
    if u.scheme not in {"http", "https"} or not u.hostname:
        raise ValueError(f"MLflow tracking_uri inválido: {uri}")
    try:
        socket.gethostbyname(u.hostname)
    except socket.gaierror as e:
        raise ValueError(f"Não consegui resolver host '{u.hostname}' em tracking_uri={uri}") from e

def publish_to_mlflow(tracking_uri: str, params: Dict) -> str:
    """
    Publica o modelo no MLflow e promove para 'Production', lendo tudo de params:deploy.
    Espera em params:
      - mlflow_experiment (str)
      - mlflow_model_name (str)
      - latest_model_path (str)
    """
    tracking_uri = (tracking_uri or "").strip()
    _validate_tracking_uri(tracking_uri)

    latest_model_path = params.get("latest_model_path")
    if not latest_model_path:
        raise ValueError("Parâmetro 'latest_model_path' ausente em params:deploy.")
    model_path = Path(latest_model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    experiment_name = params.get("mlflow_experiment", "default")
    model_name = params.get("mlflow_model_name", "model")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow URI: %s | Experimento: %s | Modelo: %s", tracking_uri, experiment_name, model_name)
    logger.info("Publicando arquivo: %s", model_path)

    with mlflow.start_run():
        #mlflow.log_artifact(str(model_path), artifact_path="model")
        model = joblib.load(model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"Nenhuma versão encontrada para '{model_name}' após o log_model().")
    latest_version = max(versions, key=lambda v: int(v.version))

    logger.info("Promovendo '%s' versão %s para Production…", model_name, latest_version.version)
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    model_uri = f"models:/{model_name}/Production"
    logger.info("✅ Modelo publicado em: %s", model_uri)
    return model_uri
