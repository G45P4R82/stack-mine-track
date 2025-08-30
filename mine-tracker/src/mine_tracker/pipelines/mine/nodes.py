"""
This is a boilerplate pipeline 'mine'
generated using Kedro 1.0.0
"""
import pandas as pd

def carregar_dados():
    """Carrega o CSV e converte timestamp corretamente."""
    caminho_csv = "https://dl.minetrack.me/Java/1-8-2021.csv"
    df = pd.read_csv(caminho_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def gerar_features(df):
    """Cria todas as features para análise."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Features temporais
    df['data'] = df['timestamp'].dt.date
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    df['dia_da_semana'] = df['timestamp'].dt.day_name()
    df['final_de_semana'] = df['dia_da_semana'].isin(['Saturday', 'Sunday']).astype(int)

    # Variação e tendência
    df['var_jogadores'] = df.groupby('ip')['playerCount'].diff()
    df['pct_var_jogadores'] = df.groupby('ip')['playerCount'].pct_change() * 100
    df['media_movel_10'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df['media_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['desvio_movel_30'] = df.groupby('ip')['playerCount'].transform(lambda x: x.rolling(window=30, min_periods=1).std())

    # Popularidade relativa
    df['total_jogadores'] = df.groupby('timestamp')['playerCount'].transform('sum')
    df['proporcao_rede'] = df['playerCount'] / df['total_jogadores']

    # Flags de eventos
    limite_pico = df['playerCount'].quantile(0.95)
    df['flag_pico'] = (df['playerCount'] > limite_pico).astype(int)
    df['queda_abrupta'] = (df['pct_var_jogadores'] < -20).astype(int)
    df['recuperacao'] = (df['pct_var_jogadores'] > 20).astype(int)

    # Ciclos de demanda
    df['periodo_dia'] = pd.cut(
        df['hora'],
        bins=[0, 6, 12, 18, 24],
        labels=['Madrugada', 'Manhã', 'Tarde', 'Noite'],
        right=False
    )

    # Intervalos entre registros
    df['intervalo_segundos'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds()

    # Codificação para ML
    df['server_id'] = df['ip'].astype('category').cat.codes
    df['servidor_hora'] = df['ip'] + "_" + df['hora'].astype(str)

    return df

def salvar_dados(df, caminho_saida):
    """Salva o dataset enriquecido em CSV."""
    df.to_csv(caminho_saida, index=False)
    print(f"Dataset salvo em: {caminho_saida}")

def main():
    entrada = "https://dl.minetrack.me/Java/1-8-2021.csv"  # caminho do CSV original
    saida = "minecraft_servidores_features.csv"

    print("Carregando dados...")
    df = carregar_dados(entrada)

    print("Gerando features...")
    df = gerar_features(df)

    print("Salvando dataset enriquecido...")
    salvar_dados(df, saida)

    print("Processo concluído!")

if __name__ == "__main__":
    main()
