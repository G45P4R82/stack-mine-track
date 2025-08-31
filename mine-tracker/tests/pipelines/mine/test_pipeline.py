"""
This is a boilerplate test file for pipeline 'mine'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
from unittest.mock import patch
import pandas as pd

from mine_tracker.pipelines.mine.nodes import carregar_dados, gerar_features


def test_carregar_dados():
    """Testa se carregar_dados funciona"""
    with patch('pandas.read_csv') as mock_read:
        # Mock simples
        mock_read.return_value = pd.DataFrame({
            'ip': ['192.168.1.1'],
            'playerCount': [100],
            'timestamp': ['1640995200000']
        })
        
        result = carregar_dados()
        
        # Só verifica se retorna algo e não quebra
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


def test_gerar_features():
    """Testa se gerar_features funciona"""
    # Dados de teste simples
    df_teste = pd.DataFrame({
        'ip': ['192.168.1.1', '192.168.1.1'],
        'playerCount': [100, 120],
        'timestamp': pd.to_datetime(['2021-01-01 10:00:00', '2021-01-01 10:01:00'])
    })
    
    result = gerar_features(df_teste)
    
    # Só verifica se retorna algo e não quebra
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'hora' in result.columns  # Uma feature básica
