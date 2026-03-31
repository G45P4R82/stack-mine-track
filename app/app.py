from flask import Flask, render_template, jsonify, Response
import json
import os
from collections import defaultdict

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'mine-tracker', 'data', '08_reporting')


def load_metricas():
    path = os.path.join(DATA_DIR, 'metricas.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def build_summary():
    """Pré-processa o report_inference.json e devolve um resumo compacto."""
    json_path = os.path.join(DATA_DIR, 'report_inference.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.loads(f.read().replace(': NaN', ': null'))
    except Exception:
        return {}

    summary_clusters = []
    for cluster in raw.get('clusters', []):
        instances = cluster.get('instances', [])

        # --- agregar por hora ---
        hourly = defaultdict(lambda: {'playerCount': [], 'media_movel_10': [], 'target_24h': []})
        last_instance = {}
        for inst in instances:
            h = inst.get('hora', 0)
            if h is None:
                h = 0
            h = int(h)
            pc = inst.get('playerCount')
            if pc is not None:
                hourly[h]['playerCount'].append(pc)
            mm = inst.get('media_movel_10')
            if mm is not None:
                hourly[h]['media_movel_10'].append(mm)
            t24 = inst.get('target_24h')
            if t24 is not None:
                hourly[h]['target_24h'].append(t24)
            last_instance = inst  # a última instância no array

        hourly_summary = {}
        for h, vals in sorted(hourly.items()):
            hourly_summary[h] = {
                'avg_playerCount': round(sum(vals['playerCount']) / len(vals['playerCount']), 1) if vals['playerCount'] else None,
                'avg_media_movel_10': round(sum(vals['media_movel_10']) / len(vals['media_movel_10']), 1) if vals['media_movel_10'] else None,
                'avg_target_24h': round(sum(vals['target_24h']) / len(vals['target_24h']), 1) if vals['target_24h'] else None,
                'count': len(vals['playerCount']),
            }

        # --- variação primeiro→último ---
        first = instances[0] if instances else {}
        pc_first = first.get('playerCount')
        pc_last = last_instance.get('playerCount')
        if pc_first and pc_last and pc_first != 0:
            variacao_pct = round(((pc_last - pc_first) / pc_first) * 100, 2)
        else:
            variacao_pct = 0

        summary_clusters.append({
            'domain': cluster.get('domain'),
            'cluster_id': cluster.get('cluster_id'),
            'baseline_prediction': cluster.get('baseline_prediction'),
            'level': cluster.get('level'),
            'action': cluster.get('action'),
            'server_mean': last_instance.get('server_mean'),
            'media_movel_10': last_instance.get('media_movel_10'),
            'last_timestamp': last_instance.get('timestamp'),
            'variacao_pct': variacao_pct,
            'hourly': hourly_summary,
        })

    return {
        'legend': raw.get('legend', {}),
        'clusters': summary_clusters,
        'metricas': load_metricas(),
    }


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard_alt():
    return render_template('dashboard.html')


@app.route('/api/data')
def api_data():
    summary = build_summary()
    return Response(json.dumps(summary, ensure_ascii=False), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8901, debug=True)