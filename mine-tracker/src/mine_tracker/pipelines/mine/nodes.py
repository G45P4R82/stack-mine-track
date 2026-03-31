"""
Pipeline 'mine' — Coleta de dados e Feature Engineering.
"""
import logging
import numpy as np
import random
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
import shutil

import pandas as pd
import requests
import pytz

import mlflow

# MLflow config
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("minecraft_mine_tracker")

logger = logging.getLogger(__name__)


def carregar_dados(edition: str = "Java", days_history: int = 60) -> pd.DataFrame:
    """Baixa CSVs do período especificado do minetrack.me salvando em disco.

    Args:
        edition: Edição do Minecraft ("Java" ou "Bedrock").
        days_history: Quantidade de dias para buscar dados para o passado a partir de hoje.

    Returns:
        DataFrame consolidado.
    """
    hoje = datetime.today()
    fim = hoje - timedelta(days=1)  # Até ontem (pois hoje pode estar incompleto)
    inicio = hoje - timedelta(days=days_history)

    # Criar diretório temporário para os CSVs diários
    temp_dir = Path("data/01_raw/temp_collect")
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Coletando dados de %s — período: %s até %s (Salvando em disk)",
        edition,
        inicio.strftime("%d/%m/%Y"),
        fim.strftime("%d/%m/%Y"),
    )

    temp_files = []
    dia = inicio
    while dia <= fim:
        url = f"https://dl.minetrack.me/{edition}/{dia.day}-{dia.month}-{dia.year}.csv"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            temp_path = temp_dir / f"{dia.strftime('%Y-%m-%d')}.csv"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            temp_files.append(temp_path)
            logger.info("✅ Baixado e salvo em disco: %s", url)
        except Exception as e:
            logger.warning("⚠️ Erro ao baixar %s: %s", url, e)
        dia += timedelta(days=1)

    if not temp_files:
        logger.error("Nenhum dado foi coletado.")
        return pd.DataFrame()

    # Junta tudo (Lendo em disco para evitar estourar memória)
    logger.info("Iniciando merge dos %d arquivos coletados...", len(temp_files))
    
    # Implementação de merge linear simples para retornar como DataFrame 
    # (O Kedro precisa do objeto final em memória para o CSVDataset salvar)
    dfs = []
    for path in temp_files:
        dfs.append(pd.read_csv(path))
    
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Conversão de timestamp pós-concatenação para economizar tempo
    if not final_df.empty:
        final_df["timestamp"] = pd.to_datetime(
            final_df["timestamp"], unit="ms", errors="coerce", utc=True
        )

    # Limpeza
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Consolidação concluída. Memória liberada.")

    return final_df


def carregar_dados_ultimas_4h(edition: str = "Java") -> pd.DataFrame:
    """
    Baixa o CSV de ontem do minetrack.me para ser usado na inferência.
    """
    ontem = datetime.today() - timedelta(days=1)
    url = f"https://dl.minetrack.me/{edition}/{ontem.day}-{ontem.month}-{ontem.year}.csv"
    
    logger.info("Coletando dados de inferência (ontem): %s", url)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], unit="ms", errors="coerce", utc=True
        )
        
        # Para manter compatibilidade com o pipeline de inference/report, 
        # gera uma coluna 'cluster' baseada no código do IP caso não exista.
        if "cluster" not in df.columns:
            df["cluster"] = df["ip"].astype("category").cat.codes
            
        logger.info("✅ Dados de inferência carregados: %d registros.", len(df))
        return df
    except Exception as e:
        logger.error("❌ Erro ao coletar dados de inferência: %s", e)
        return pd.DataFrame()


def gerar_features(df):
    """Cria todas as features para análise."""
    df = df.copy()
    logger.info("Gerando features a partir dos dados brutos.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Features temporais
    logger.info("Criando features temporais.")
    df['data'] = df['timestamp'].dt.date
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['dia_da_semana_num'] = df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    df['dia_da_semana'] = df['timestamp'].dt.day_name()
    df['final_de_semana'] = df['dia_da_semana'].isin(['Saturday', 'Sunday']).astype(int)

    # ===== CODIFICAÇÃO CÍCLICA (sin/cos) para hora e dia da semana =====
    logger.info("Codificando hora e dia da semana como features cíclicas (sin/cos).")
    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
    df['dia_sem_sin'] = np.sin(2 * np.pi * df['dia_da_semana_num'] / 7)
    df['dia_sem_cos'] = np.cos(2 * np.pi * df['dia_da_semana_num'] / 7)

    # ===== LAG FEATURES (informação que EXISTE no momento da previsão) =====
    logger.info("Criando features de lag (lag_24h, lag_1h).")
    # Lag de 24h: o valor que o servidor tinha exatamente 24h atrás
    # Usando shift por grupo — cada tick é ~3min, 24h ≈ 480 ticks
    df = df.sort_values(['ip', 'timestamp'])
    df['playerCount_lag_24h'] = df.groupby('ip')['playerCount'].shift(480)
    # Lag de 1h (~20 ticks)
    df['playerCount_lag_1h'] = df.groupby('ip')['playerCount'].shift(20)

    # Variação e tendência
    logger.info("Calculando variações e médias móveis.")
    df['var_jogadores'] = df.groupby('ip')['playerCount'].diff()
    df['pct_var_jogadores'] = df.groupby('ip')['playerCount'].pct_change(fill_method=None) * 100
    df['media_movel_10'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['media_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['desvio_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

    # ===== NORMALIZAÇÃO POR SERVIDOR (Z-Score por IP) =====
    logger.info("Normalizando playerCount por servidor (z-score por IP).")
    server_stats = df.groupby('ip')['playerCount'].agg(['mean', 'std']).rename(columns={'mean': 'server_mean', 'std': 'server_std'})
    df = df.merge(server_stats, on='ip', how='left')
    df['server_std'] = df['server_std'].replace(0, 1)  # evita divisão por zero
    df['playerCount_z'] = (df['playerCount'] - df['server_mean']) / df['server_std']
    df['lag_24h_z'] = (df['playerCount_lag_24h'] - df['server_mean']) / df['server_std']

    # Popularidade relativa
    logger.info("Calculando popularidade relativa.")
    df['total_jogadores'] = df.groupby('timestamp')['playerCount'].transform('sum')
    df['proporcao_rede'] = df['playerCount'] / df['total_jogadores']

    # Flags de eventos
    logger.info("Criando flags de eventos especiais.")
    limite_pico = df['playerCount'].quantile(0.95)
    df['flag_pico'] = (df['playerCount'] > limite_pico).astype(int)
    df['queda_abrupta'] = (df['pct_var_jogadores'] < -20).astype(int)
    df['recuperacao'] = (df['pct_var_jogadores'] > 20).astype(int)

    # Ciclos de demanda
    logger.info("Adicionando ciclos de demanda.")
    df['periodo_dia'] = pd.cut(
        df['hora'],
        bins=[0, 6, 12, 18, 24],
        labels=['Madrugada', 'Manhã', 'Tarde', 'Noite'],
        right=False
    )

    # Intervalos entre registros
    logger.info("Calculando intervalos entre registros.")
    df['intervalo_segundos'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds()

    # Codificação para ML
    logger.info("Codificando variáveis categóricas.")
    df['server_id'] = df['ip'].astype('category').cat.codes
    df['servidor_hora'] = df['ip'] + "_" + df['hora'].astype(str)

    # ===== D+1 DIRECT FORECASTING: TARGET 24H =====
    logger.info("Criando target_24h para previsão D+1.")
    
    # Remover NaT que quebram o merge_asof
    df = df.dropna(subset=['timestamp'])
    
    df_future = df[['ip', 'timestamp', 'playerCount']].copy()
    df_future['timestamp_match'] = df_future['timestamp'] - pd.Timedelta(hours=24)
    df_future = df_future.rename(columns={'playerCount': 'target_24h'})
    
    # merge_asof REQUIRE que os dataframes estejam estritamente ordenados pela chave principal
    df = df.sort_values(by='timestamp')
    df_future = df_future.sort_values(by='timestamp_match')
    
    df = pd.merge_asof(
        df,
        df_future.drop(columns='timestamp'),
        left_on='timestamp',
        right_on='timestamp_match',
        by='ip',
        direction='nearest',
        tolerance=pd.Timedelta(minutes=30)
    ).drop(columns=['timestamp_match'])

    try:
        gerar_graficos_distribuicao(df)
    except Exception as e:
        logger.warning(f"Erro ao gerar gráficos de distribuição: {e}")

    # Loga metadados do processamento de dados no MLflow
    with mlflow.start_run(run_name="feature_engineering_run", nested=True):
        mlflow.log_param("num_servers", df['ip'].nunique())
        mlflow.log_metric("total_rows", len(df))
        mlflow.log_metric("nan_ratio", df.isna().sum().sum() / df.size)
        mlflow.set_tag("phase", "feature_engineering")

    return df

def gerar_graficos_distribuicao(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os
    
    logger.info("Gerando gráficos de distribuição das features...")
    os.makedirs("data/04_feature/plots", exist_ok=True)
    
    # 1. Plot histograma target
    if 'target_24h' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['target_24h'].dropna(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.title('Distribuição da Variável Alvo (target_24h)')
        plt.xlabel('Número de Jogadores (D+1)')
        plt.ylabel('Frequência (horas iteradas)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("data/04_feature/plots/distribuicao_target.png", dpi=200)
        plt.close()
        
    # 2. Plot correlação simples heatmap via matshow
    cols_to_plot = ["playerCount", "media_movel_10", "proporcao_rede", "pct_var_jogadores"]
    if 'target_24h' in df.columns:
        cols_to_plot.append("target_24h")
        
    cols_present = [c for c in cols_to_plot if c in df.columns]
    
    if len(cols_present) > 1:
        corr = df[cols_present].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(cols_present)))
        ax.set_yticks(range(len(cols_present)))
        ax.set_xticklabels(cols_present, rotation=45, ha='left')
        ax.set_yticklabels(cols_present)
        
        for i in range(len(cols_present)):
            for j in range(len(cols_present)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr.iloc[i, j]) < 0.5 else 'white')
                
        plt.title('Correlação entre Features e Target (D+1)', pad=20)
        plt.tight_layout()
        plt.savefig("data/04_feature/plots/correlacao_features.png", dpi=200)
        plt.close()

    # 3. Série temporal top server
    if 'ip' in df.columns and 'timestamp' in df.columns:
        top_server = df['ip'].value_counts().idxmax()
        subset = df[df['ip'] == top_server].sort_values('timestamp').tail(1000)
        plt.figure(figsize=(12, 5))
        plt.plot(subset['timestamp'], subset['playerCount'], label='Atual (Raw)', color='green')
        if 'target_24h' in subset.columns:
            plt.plot(subset['timestamp'], subset['target_24h'], label='Target D+1', color='red', alpha=0.5, linestyle='--')
        plt.title(f'Série Temporal Recente (últimos 1000 ticks) - {top_server}')
        plt.xlabel('Tempo')
        plt.ylabel('Jogadores')
        plt.legend()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.gcf().autofmt_xdate()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("data/04_feature/plots/serie_temporal_top_server.png", dpi=200)
        plt.close()
    
    logger.info("Gráficos de análise de features salvos com sucesso.")
    