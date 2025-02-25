import os
import json
import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import logging
import time

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'votre_clé_secrète'  # Remplacez par une valeur sécurisée
socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def read_json_file(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"error": f"Erreur de décodage dans le fichier {filepath}"}
    return {"error": f"Fichier non trouvé: {filepath}"}

def get_metrics_data():
    # Récupération des métriques depuis les fichiers logs
    performance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/performance.json')
    trades_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/trade_history.json')
    performance_chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/performance_chart.json')
    risk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/risk_metrics.json')
    
    data = {
        "performance": read_json_file(performance_path),
        "trades": read_json_file(trades_path),
        "performance_chart": read_json_file(performance_chart_path),
        "risk": read_json_file(risk_path)
    }
    return data

def background_thread():
    while True:
        data = get_metrics_data()
        socketio.emit('update_metrics', data, namespace='/dashboard')
        time.sleep(10)

@app.route('/api/metrics')
def metrics():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/performance.json')
    data = read_json_file(filepath)
    return jsonify(data)

@app.route('/api/trades')
def trades():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/trade_history.json')
    data = read_json_file(filepath)
    return jsonify(data)

@app.route('/api/performance')
def performance():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/performance_chart.json')
    data = read_json_file(filepath)
    return jsonify(data)

@app.route('/api/risk')
def risk():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/risk_metrics.json')
    data = read_json_file(filepath)
    return jsonify(data)

@app.route('/api/export')
def export_data():
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/trade_history.json')
    data = read_json_file(filepath)
    return json.dumps(data)

@app.route('/')
def index():
    return render_template('dashboard.html')

@socketio.on('connect', namespace='/dashboard')
def on_connect():
    logging.info("Client connecté sur /dashboard")
    emit('update_metrics', get_metrics_data())

if __name__ == '__main__':
    socketio.start_background_task(background_thread)
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
