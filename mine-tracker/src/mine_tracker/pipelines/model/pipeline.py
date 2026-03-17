from kedro.pipeline import Node, Pipeline
from mine_tracker.pipelines.model.nodes import (
    criar_pipelines, treinar_modelos_incremental, avaliar_modelos
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=criar_pipelines,
            inputs=[
                "params:model.features",
                "params:model.training.n_estimators", 
                "params:model.training.n_jobs"
            ],
            outputs="modelos",
            name="criar_pipelines_node",
        ),
        Node(
            func=treinar_modelos_incremental,
            inputs=[
                "modelos", 
                "params:model.features", 
                "params:model.target",
                "params:model.training.n_estimators"
            ],
            outputs="modelos_treinados",
            name="treinar_modelos_node",
        ),
        Node(
            func=avaliar_modelos,
            inputs=[
                "modelos_treinados", 
                "params:model.features", 
                "params:model.target"
            ],
            outputs=["best_model", "metricas_dict", "X_test"],
            name="avaliar_modelos_node",
        ),
    ])
