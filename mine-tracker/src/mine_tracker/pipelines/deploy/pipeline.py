from kedro.pipeline import Pipeline, node
from mine_tracker.pipelines.deploy.nodes import publish_to_mlflow

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=publish_to_mlflow,
            inputs=["mlflow_tracking_uri", "params:deploy"],
            outputs="mlflow_model_uri",
            name="publish_to_mlflow_node",
        ),
    ])