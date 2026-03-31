"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.0.0
"""

from typing import Dict, Any
import pandas as pd
import logging
import joblib

from mine_tracker.pipelines.mine.nodes import gerar_features
import numpy as np
import matplotlib.pyplot as plt
import os

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

    # Gera features a partir dos dados brutos
    df_feat = gerar_features(df)
    
    # É essencial que a ordem seja a mesma do treinamento
    features_cols = [
        "hora_sin",
        "hora_cos",
        "dia_sem_sin",
        "dia_sem_cos",
        "final_de_semana",
        "playerCount_lag_24h",
        "playerCount_lag_1h",
        "playerCount_z",
        "lag_24h_z",
        "proporcao_rede",
    ]
    
    # Trata as features do mesmo modo que no treino
    X_pred = df_feat[features_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Faz predição
    df_feat["prediction"] = model.predict(X_pred)

    logger.info("Inferência concluída. Gerando gráficos da previsão de D+1...")
    try:
        gerar_grafico_previsao(df_feat)
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos: {e}")

    return df_feat

def gerar_grafico_previsao(df: pd.DataFrame) -> None:
    import matplotlib.dates as mdates
    
    # A previsão é para D+1 (timestamp + 24h)
    df_plot = df.copy()
    if 'timestamp' not in df_plot.columns:
        return
        
    df_plot['timestamp_previsao'] = df_plot['timestamp'] + pd.Timedelta(hours=24)
    
    os.makedirs("data/08_reporting", exist_ok=True)
    
    # Identifica o servidor ou top servidores
    if 'ip' in df_plot.columns:
        top_clusters = df_plot.groupby('ip')['prediction'].mean().nlargest(5).index
        
        plt.figure(figsize=(14, 8))
        for ip in top_clusters:
            subset = df_plot[df_plot['ip'] == ip].sort_values('timestamp_previsao')
            plt.plot(subset['timestamp_previsao'], subset['prediction'], linewidth=1.5, label=ip)
            
        plt.title('Previsão de Jogadores p/ as Próximas 24h (Top 5 Servidores)', fontsize=14, weight='bold')
        plt.xlabel('Horário Previsto (D+1)', fontsize=12)
        plt.ylabel('Jogadores (Previsto)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        info_text_top5 = (
            "💡 GUIA DE DECISÃO:\n"
            "- Fique alerta quando vários servidores atingem pico simultâneo (risco de gargalo sistêmico).\n"
            "- Aumentos constantes da média podem exigir Load Balancing."
        )
        plt.figtext(0.13, 0.02, info_text_top5, fontsize=11, bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()
        
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("data/08_reporting/previsao_24h_top5.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        for ip in top_clusters:
            plt.figure(figsize=(12, 7))
            subset = df_plot[df_plot['ip'] == ip].sort_values('timestamp_previsao').reset_index(drop=True)
            plt.plot(subset['timestamp_previsao'], subset['prediction'], color='purple', linewidth=2, label='Prévia de Demanda')
            plt.fill_between(subset['timestamp_previsao'], subset['prediction'], alpha=0.2, color='purple')
            
            # Encontra o pico máximo
            if not subset.empty:
                max_idx = subset['prediction'].idxmax()
                max_time = subset.loc[max_idx, 'timestamp_previsao']
                max_val = subset.loc[max_idx, 'prediction']
                
                ax = plt.gca()
                ax.annotate(
                    'Pico de Demanda Esperado\n(alocar +recursos)',
                    xy=(max_time, max_val),
                    xytext=(0, 40), textcoords='offset points',
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="r", lw=1, alpha=0.8),
                    fontsize=10, color='red', weight='bold', ha='center'
                )
                
                # Linha de média
                avg_val = subset['prediction'].mean()
                ax.axhline(avg_val, color='orange', linestyle='--', label=f'Média de Jogadores: {avg_val:.0f}')
            
            info_text = (
                "COMO LER ESTE GRÁFICO:\n"
                "• A linha roxa mostra a expectativa de jogadores. A faixa indica o volume acumulado.\n"
                "• Linha acima da média laranja -> Volume alto, prepare escalonamento vertical.\n"
                "• Quedas prolongadas abaixo da média -> Janela ideal e segura para manutenção."
            )
            plt.figtext(0.13, 0.02, info_text, fontsize=10, bbox=dict(facecolor='whitesmoke', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
            
            plt.title(f'Previsão de Demanda 24h - Servidor: {ip}', fontsize=14, weight='bold')
            plt.xlabel('Horário Previsto (D+1)', fontsize=12)
            plt.ylabel('Jogadores (Previsto)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')
            
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gcf().autofmt_xdate()
            
            plt.subplots_adjust(bottom=0.25)
            nome_arquivo = ip.replace(".", "_").replace(":", "_")
            plt.savefig(f"data/08_reporting/previsao_24h_{nome_arquivo}.png", dpi=300, bbox_inches='tight')
            plt.close()




def generate_report(df: pd.DataFrame) -> Dict[str, Any]:
    # 🔹 legenda explicando cada feature
    legend = {
        "hora": "Hora do dia (0–23)",
        "final_de_semana": "Indicador se é fim de semana (0=Não, 1=Sim)",
        "playerCount": "Número de jogadores no instante",
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
        domain_name = subset['ip'].iloc[0] if not subset.empty and 'ip' in subset.columns else f"Cluster {cluster_id}"
        
        # Converte NaN para None para gerar JSON válido sem erros no JS posteriormente
        subset_clean = subset.replace({np.nan: None})

        cluster_info = {
            "domain": domain_name,
            "cluster_id": cluster_id,
            "baseline_prediction": pred,
            "level": level,
            "action": action,
            "instances": subset_clean.to_dict(orient="records")
        }
        report["clusters"].append(cluster_info)

    for i, row in rank.iterrows():
        pred = round(float(row["prediction"]))
        lvl = label_load(pred)
        cluster_id = int(row["cluster"])
        
        subset = df[df["cluster"] == cluster_id]
        domain_name = subset['ip'].iloc[0] if not subset.empty and 'ip' in subset.columns else f"Cluster {cluster_id}"

        report["ranking"].append({
            "posicao": i + 1,
            "domain": domain_name,
            "cluster_id": cluster_id,
            "prediction": pred,
            "level": lvl
        })

    return report
