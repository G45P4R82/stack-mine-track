from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

# Carregar conteúdo do JSON corrigindo inconsistências (NaN)
def get_clean_json_string():
    json_path = os.path.join(os.path.dirname(__file__), '..', 'mine-tracker', 'data', '08_reporting', 'report_inference.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return f.read().replace(': NaN', ': null')
    except Exception as e:
        return "{}"

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard_alt():
    return render_template('dashboard.html')

@app.route('/api/data')
def api_data():
    from flask import Response
    content = get_clean_json_string()
    return Response(content, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8901, debug=True)