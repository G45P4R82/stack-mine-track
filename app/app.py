# -*- coding: utf-8 -*-
# app.py que vai fazer load do modelo e fazer a predição usando flask framework

import flask
import mlflow


app = flask.Flask(__name__)

def load_model(params):
    run_uri = "runs:/260ba19b0ca84646bee0b3fc4663099e/minecraft-model"
    mlflow.set_tracking_uri(params.get("uri"))
    mlflow.set_experiment(params.get("experiment_name"))
    model = mlflow.sklearn.load_model(run_uri)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


@app.route('/predict', methods=['POST'])
def predict():
    params = flask.request.json
    prediction = predict(model, params.get("X_test"))
    return flask.jsonify({"prediction": prediction})  

if __name__ == '__main__':
    params = {"uri": "http://192.168.0.43:3000", "experiment_name": "minecraft-training"}
    model = load_model(params)
    app.run(host='0.0.0.0', port=8901, debug=True)