"""
Testes para o pipeline 'mine'.
"""
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta
import pandas as pd

from mine_tracker.pipelines.mine.nodes import carregar_dados, gerar_features


def test_carregar_dados_usa_mes_anterior():
    """Testa se carregar_dados calcula o mês anterior e baixa CSVs."""
    with patch('mine_tracker.pipelines.mine.nodes.pd.read_csv') as mock_read:
        mock_read.return_value = pd.DataFrame({
            'ip': ['192.168.1.1'],
            'playerCount': [100],
            'timestamp': ['1640995200000']
        })

        result = carregar_dados(edition="Java")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Verifica que as URLs chamadas correspondem ao mês anterior
        hoje = datetime.today()
        primeiro_dia_mes_atual = hoje.replace(day=1)
        ultimo_dia_mes_anterior = primeiro_dia_mes_atual - timedelta(days=1)

        chamadas = [str(call) for call in mock_read.call_args_list]
        # Verifica que o mês/ano do mês anterior aparece nas chamadas
        mes_anterior = ultimo_dia_mes_anterior.month
        ano_anterior = ultimo_dia_mes_anterior.year
        assert any(f"-{mes_anterior}-{ano_anterior}" in c for c in chamadas), (
            f"Esperava URLs com mês {mes_anterior}/{ano_anterior}, mas chamadas foram: {chamadas}"
        )


def test_carregar_dados_retorna_vazio_se_nenhum_csv():
    """Testa se retorna DataFrame vazio quando todos os CSVs falham."""
    with patch('mine_tracker.pipelines.mine.nodes.pd.read_csv') as mock_read:
        mock_read.side_effect = Exception("404 Not Found")

        result = carregar_dados(edition="Java")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def test_gerar_features():
    """Testa se gerar_features funciona."""
    df_teste = pd.DataFrame({
        'ip': ['192.168.1.1', '192.168.1.1'],
        'playerCount': [100, 120],
        'timestamp': pd.to_datetime(['2021-01-01 10:00:00', '2021-01-01 10:01:00'])
    })

    result = gerar_features(df_teste)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'hora' in result.columns
    assert 'final_de_semana' in result.columns
    assert 'media_movel_10' in result.columns
