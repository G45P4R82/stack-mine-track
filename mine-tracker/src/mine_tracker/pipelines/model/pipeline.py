from kedro.pipeline import Node, Pipeline
from mine_tracker.pipelines.model.nodes import (
    load_data, preprocess_data, criar_pipelines, treinar_modelos, avaliar_modelos
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=criar_pipelines,
            inputs=None,
            outputs="modelos",
            name="criar_pipelines_node",
        ),
        Node(
            func=load_data,
            inputs="minecraft_servidores_features",
            outputs="model_data",
            name="load_data_node",
        ),
        Node(
            func=preprocess_data,
            inputs="model_data",
            outputs=["X", "y", "n_drop_y"],
            name="preprocess_data_node",
        ),
        Node(  # <- AGORA PRODUZ "modelos_trained"
            func=treinar_modelos,
            inputs=["modelos", "X", "y"],
            outputs="modelos_trained",
            name="treinar_modelos_node",
        ),
        Node(  # <- AVALIA USA "modelos_trained"
            func=avaliar_modelos,
            inputs=["modelos_trained", "X", "y", "n_drop_y"],
            outputs=["best_model", "metricas_dict", "X_test"],
            name="avaliar_modelos_node",
        ),
    ])
