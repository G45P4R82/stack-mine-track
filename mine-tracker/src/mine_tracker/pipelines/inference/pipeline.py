"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from mine_tracker.pipelines.inference.nodes import inferencia, generate_report

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=inferencia,
            inputs=["best_model", "input_inference"],
            outputs="output_inference",
            name="inferencia_node",
        ),
        Node(
            func=generate_report,
            inputs="output_inference",
            outputs="report_inference",
            name="generate_report_node",
        ),
    ])
