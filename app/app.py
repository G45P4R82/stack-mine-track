from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

# Carregar dados do JSON
def load_data():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'mine-tracker', 'data', '08_reporting', 'report_inference.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/')
def dashboard():
    data = load_data()
    return render_template('dashboard.html', data=data)

@app.route('/dashboard')
def dashboard_alt():
    data = load_data()
    return render_template('dashboard.html', data=data)

@app.route('/api/data')
def api_data():
    data = load_data()
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8901, debug=True)