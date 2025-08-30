from kedro.pipeline import Pipeline, node, pipeline
from .nodes import carregar_dados, gerar_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=carregar_dados,
            inputs=None,   # <— usa o dataset do catálogo
            outputs="minecraft_servidores_raw",
            name="coleta_mine_node",
        ),
        node(
            func=gerar_features,
            inputs="minecraft_servidores_raw",
            outputs="minecraft_servidores_features",
            name="coleta_mine_node_features",
        ),
    ])
