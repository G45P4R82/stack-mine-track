"""
Pipeline 'mine' — Coleta de dados e Feature Engineering.
"""
import logging
import random
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
import shutil

import pandas as pd
import requests
import pytz

logger = logging.getLogger(__name__)


def carregar_dados(edition: str = "Java") -> pd.DataFrame:
    """Baixa CSVs do mês anterior completo do minetrack.me salvando em disco.

    Args:
        edition: Edição do Minecraft ("Java" ou "Bedrock").

    Returns:
        DataFrame consolidado.
    """
    hoje = datetime.today()
    primeiro_dia_mes_atual = hoje.replace(day=1)
    ultimo_dia_mes_anterior = primeiro_dia_mes_atual - timedelta(days=1)
    primeiro_dia_mes_anterior = ultimo_dia_mes_anterior.replace(day=1)

    inicio = primeiro_dia_mes_anterior
    fim = ultimo_dia_mes_anterior

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
    df['dia_da_semana'] = df['timestamp'].dt.day_name()
    df['final_de_semana'] = df['dia_da_semana'].isin(['Saturday', 'Sunday']).astype(int)

    # Variação e tendência
    logger.info("Calculando variações e médias móveis.")
    df['var_jogadores'] = df.groupby('ip')['playerCount'].diff()
    df['pct_var_jogadores'] = df.groupby('ip')['playerCount'].pct_change() * 100
    df['media_movel_10'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['media_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['desvio_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

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

    return df
