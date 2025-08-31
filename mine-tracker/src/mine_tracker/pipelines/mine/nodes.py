"""
This is a boilerplate pipeline 'mine'
generated using Kedro 1.0.0
"""
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import random
from datetime import datetime
import pytz
from datetime import datetime, timedelta


def carregar_dados():
    """Carrega os CSVs de 02/04/2020 até 30/07/2020 (Java edition) e concatena."""
    inicio = datetime(2022, 4, 2)
    fim = datetime(2022, 4, 10)
    dados = []

    dia = inicio
    while dia <= fim:
        url = f"https://dl.minetrack.me/Java/{dia.day}-{dia.month}-{dia.year}.csv"
        try:
            df = pd.read_csv(url)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            dados.append(df)
            print(f"✅ carregado {url}")
        except Exception as e:
            print(f"⚠️ erro em {url}: {e}")
        dia += timedelta(days=1)

    if dados:
        return pd.concat(dados, ignore_index=True)
    else:
        return pd.DataFrame()


def carregar_dados_ultimas_4h() -> pd.DataFrame:
    """
    Simula a coleta de dados coerente com clusters conhecidos,
    usando horário do Brasil e variação de ±1h.
    """
    tz = pytz.timezone("America/Sao_Paulo")
    now = datetime.now(tz)

    # clusters e horários base
    clusters_config = {
        0: {"hora_base": 10, "media_range": (58000, 62000)},
        1: {"hora_base": 18, "media_range": (68000, 72000)},
        2: {"hora_base": 21, "media_range": (87000, 91000)},
        3: {"hora_base": 3,  "media_range": (38000, 42000)},
    }

    rows = []
    for cluster_id, cfg in clusters_config.items():
        hora = cfg["hora_base"] + random.choice([-1, 0, 1])
        hora = max(0, min(23, hora))  # garante 0–23
        final_de_semana = 1 if now.weekday() >= 5 else 0

        media_movel_10 = random.randint(*cfg["media_range"])
        proporcao_rede = round(random.uniform(0.25, 0.40), 2)
        pct_var_jogadores = round(random.uniform(-1.0, 2.0), 1)

        rows.append({
            "hora": hora,
            "final_de_semana": final_de_semana,
            "media_movel_10": media_movel_10,
            "proporcao_rede": proporcao_rede,
            "pct_var_jogadores": pct_var_jogadores,
            "cluster": cluster_id
        })

    return pd.DataFrame(rows)


def gerar_features(df):
    """Cria todas as features para análise."""
    df = df.copy()
    logger.info("Gerando features a partir dos dados brutos.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Features temporais
    logger
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


