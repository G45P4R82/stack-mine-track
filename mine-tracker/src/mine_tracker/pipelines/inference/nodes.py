"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.0.0
"""

from typing import Dict, Any
import pandas as pd
import logging
import joblib

LOAD_THRESHOLDS = {"low": 30000, "medium": 60000, "high": 90000}


def label_load(pred: float) -> str:
    if pred < LOAD_THRESHOLDS["low"]: return "baixo"
    if pred < LOAD_THRESHOLDS["medium"]: return "médio"
    if pred < LOAD_THRESHOLDS["high"]: return "alto"
    return "crítico"

def action_for_load(level: str) -> str:
    actions = {
        "baixo": "Janela boa p/ manutenção; reduzir recursos.",
        "médio": "Monitorar; ajustar autoscaling conforme tendência.",
        "alto": "Preparar autoscaling; adiar manutenção; reforçar capacidade.",
        "crítico": "Alerta de sobrecarga; ativar mitigação; limitar eventos."
    }
    return actions.get(level, "Ação não definida")


logger = logging.getLogger(__name__)

def inferencia(model, DataFrame=None, ) -> pd.DataFrame:
    if isinstance(DataFrame, dict):
        df = pd.DataFrame(DataFrame)
    elif isinstance(DataFrame, pd.DataFrame):
        df = DataFrame.copy()
    else:
        raise TypeError(f"DataFrame esperado como pandas.DataFrame ou dict, mas veio {type(DataFrame)}")

    
    # Faz predição
    df["prediction"] = model.predict(df)

    logger.info("Inferência concluída com sucesso")
    return df

def generate_report(df: pd.DataFrame) -> Dict[str, Any]:
    report = {"clusters": [], "ranking": []}

    grouped = df.groupby("cluster")["prediction"].mean().reset_index()
    rank = grouped.sort_values("prediction", ascending=False).reset_index(drop=True)

    for _, row in grouped.iterrows():
        cluster_id = int(row["cluster"])
        pred = round(float(row["prediction"]))  # 🔹 arredonda predição
        level = label_load(pred)
        action = action_for_load(level)

        subset = df[df["cluster"] == cluster_id].drop(columns=["prediction"])

        cluster_info = {
            "cluster_id": cluster_id,
            "baseline_prediction": pred,
            "level": level,
            "action": action,
            "instances": subset.to_dict(orient="records")
        }
        report["clusters"].append(cluster_info)

    for i, row in rank.iterrows():
        pred = round(float(row["prediction"]))
        lvl = label_load(pred)
        report["ranking"].append({
            "posicao": i + 1,
            "cluster_id": int(row["cluster"]),
            "prediction": pred,
            "level": lvl
        })

    return report


def generate_report(df: pd.DataFrame) -> Dict[str, Any]:
    # 🔹 legenda explicando cada feature
    legend = {
        "hora": "Hora do dia (0–23)",
        "final_de_semana": "Indicador se é fim de semana (0=Não, 1=Sim)",
        "media_movel_10": "Média móvel de jogadores nas últimas 10 janelas",
        "proporcao_rede": "Proporção de jogadores no cluster em relação à rede total (0–1)",
        "pct_var_jogadores": "Variação percentual de jogadores em relação ao período anterior"
    }

    report = {
        "legend": legend,
        "clusters": [],
        "ranking": []
    }

    grouped = df.groupby("cluster")["prediction"].mean().reset_index()
    rank = grouped.sort_values("prediction", ascending=False).reset_index(drop=True)

    for _, row in grouped.iterrows():
        cluster_id = int(row["cluster"])
        pred = round(float(row["prediction"]))  # 🔹 arredonda predição
        level = label_load(pred)
        action = action_for_load(level)

        subset = df[df["cluster"] == cluster_id].drop(columns=["prediction"])

        cluster_info = {
            "cluster_id": cluster_id,
            "baseline_prediction": pred,
            "level": level,
            "action": action,
            "instances": subset.to_dict(orient="records")
        }
        report["clusters"].append(cluster_info)

    for i, row in rank.iterrows():
        pred = round(float(row["prediction"]))
        lvl = label_load(pred)
        report["ranking"].append({
            "posicao": i + 1,
            "cluster_id": int(row["cluster"]),
            "prediction": pred,
            "level": lvl
        })

    return report
