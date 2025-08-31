import mlflow
import mlflow.sklearn

def get_Models():
    client = mlflow.MlflowClient(tracking_uri="http://192.168.0.43:3000")
    data = client.search_registered_models()
    models = [m.name for m in data]
    result = []
    for model in models:
        model_versions = {"name": model}
        data = client.search_model_versions(filter_string=f"name='{model}'", order_by=["version_number DESC"])
        versions = [dict(x) for x in data]
        model_versions["latest_versions"] = versions
        result.append(model_versions)
    return result


def test_model():
    # conecta no tracking server
    client = mlflow.MlflowClient(tracking_uri="http://192.168.0.43:3000")


    # carrega o modelo salvo como artifact desse run
    model = mlflow.sklearn.load_model(f"models:/minecraft-model/latest")

    # exemplo de input (ajuste conforme suas features)
    X_sample = [[12, 0, 45.2, 0.8, 0.12]]
    y_pred = model.predict(X_sample)

    print("Previsão:", y_pred)


if __name__ == "__main__":
    test_model()
