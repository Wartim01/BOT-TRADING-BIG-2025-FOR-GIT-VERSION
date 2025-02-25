import os
import shutil
import datetime
import time
import json
import random
import logging
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import signal  # pour la gestion des signaux

# Ajouter le dossier racine du projet dans le path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.optimization.auto_optimizer import optimize_parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration du StreamHandler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Ajout d'un FileHandler dédié aux logs du scheduler
scheduler_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/scheduler.log'))
file_handler = logging.FileHandler(scheduler_log_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def read_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Erreur de décodage dans le fichier {filepath}")
                return None
    else:
        logger.error(f"Fichier non trouvé: {filepath}")
        return None

def run_backtest():
    logger.info("Démarrage du backtest...")
    time.sleep(2)  # Simulation d'un backtest
    logger.info("Backtest terminé.")

def check_risk_management():
    logger.info("Vérification de la gestion des risques...")
    risk_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/risk_metrics.json')
    risk_data = read_json_file(risk_filepath)
    # Si le fichier est manquant ou la clé critique "max_drawdown" n'existe pas, lever une erreur
    if risk_data is None:
        logger.error("Fichier des métriques de risque manquant ou illisible. Le bot ne prendra aucune décision.")
        raise RuntimeError("Fichier risk_metrics.json manquant ou illisible.")
    if 'max_drawdown' not in risk_data:
        logger.error("Clé 'max_drawdown' manquante dans risk_metrics.json. Le bot ne prendra aucune décision.")
        raise RuntimeError("Clé 'max_drawdown' manquante dans risk_metrics.json.")
    drawdown = risk_data['max_drawdown']
    threshold = 15  # Seuil de drawdown en pourcentage
    if drawdown > threshold:
        logger.warning(f"Drawdown élevé détecté: {drawdown}% (seuil: {threshold}%)")
    else:
        logger.info(f"Drawdown dans les limites acceptables: {drawdown}%.")

def backup_logs():
    logger.info("Démarrage de la sauvegarde des logs...")
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs')
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs_backup')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = os.path.join(backup_dir, f"logs_backup_{timestamp}")
    try:
        shutil.copytree(logs_dir, destination)
        logger.info(f"Sauvegarde des logs réalisée dans {destination}.")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des logs: {e}")

def check_market_conditions():
    logger.info("Vérification des conditions de marché...")
    volatility = round(abs((datetime.datetime.now().second % 10) - 5) * 2, 2)
    threshold = 6  # Seuil de volatilité
    if volatility > threshold:
        logger.warning(f"Volatilité élevée détectée: {volatility} (seuil: {threshold})")
    else:
        logger.info(f"Volatilité normale: {volatility}.")

def run_trading():
    # Vérification préalable : le fichier risk_metrics.json doit être complet
    risk_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/risk_metrics.json')
    risk_data = read_json_file(risk_filepath)
    if risk_data is None or ('max_drawdown' not in risk_data):
        logger.error("Fichier risk_metrics.json manquant ou incomplet. Trading interrompu.")
        return

    logger.info("Exécution d'une stratégie de trading en temps réel...")
    buy_price = round(random.uniform(30000, 35000), 2)
    sell_price = round(buy_price + random.uniform(-500, 1000), 2)
    buy_fee = round(buy_price * 0.001, 2)
    sell_fee = round(sell_price * 0.001, 2)

    from src.risk import RiskManager
    risk_manager = RiskManager(account_balance=200, risk_percentage=0.01)
    stop_loss_price = round(buy_price * (1 - 0.02), 2)
    take_profit_price = round(buy_price * (1 + 0.02), 2)
    quantity = risk_manager.calculate_position_size(buy_price, stop_loss_price)

    from datetime import datetime, timedelta
    entry_timestamp = datetime.now()
    trade_delay_seconds = random.randint(60, 180)
    exit_timestamp = entry_timestamp + timedelta(seconds=trade_delay_seconds)
    trade_duration = (exit_timestamp - entry_timestamp).total_seconds()

    profit = (sell_price - buy_price) - (buy_fee + sell_fee)
    pnl_percent = round((profit / buy_price) * 100, 2)
    indicators_info = "RSI: 35, MACD: 0.5"  # À améliorer ultérieurement

    trade = {
        "id": int(time.time()),
        "asset": "BTC/USDT",
        "trade_type": "Long",
        "entry_timestamp": entry_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_timestamp": exit_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_duration_seconds": int(trade_duration),
        "entry_price": buy_price,
        "exit_price": sell_price,
        "quantity": quantity,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "pnl_amount": profit,
        "pnl_percent": pnl_percent,
        "fees": buy_fee + sell_fee,
        "status": "Gagné" if profit > 0 else "Perdu",
        "strategy": "Trend Following",
        "indicators": indicators_info,
        "confidence_score": round(random.uniform(0.7, 1.0), 2),
        "analysis": "Trade simulé.",
        "recommendation": "Aucune recommandation."
    }

    try:
        from src.notifications.notify import send_trade_pair_notification
        send_trade_pair_notification(buy_price, sell_price, buy_fee, sell_fee, debug=True)
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de la notification: {e}")

    trade_history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs/trade_history.json')
    try:
        if os.path.exists(trade_history_path):
            with open(trade_history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de l'historique des trades: {e}")
        history = []
    history.append(trade)
    try:
        with open(trade_history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info("Historique des trades mis à jour.")
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture de l'historique des trades: {e}")

def retraining_job():
    logger.info("Lancement du ré-entraîne­ment du modèle...")
    subprocess.run(["python", os.path.abspath(os.path.join("src", "training", "entrainement_lstm.py"))])
    logger.info("Ré-entraîne­ment terminé. (Note : le modèle est sauvegardé uniquement s'il est meilleur.)")

def start_scheduler():
    scheduler = BlockingScheduler()
    scheduler.add_job(run_backtest, 'interval', minutes=5, id='backtest_job')
    scheduler.add_job(check_risk_management, 'interval', minutes=3, id='risk_job')
    scheduler.add_job(backup_logs, 'interval', minutes=30, id='backup_job')
    scheduler.add_job(check_market_conditions, 'interval', minutes=2, id='market_job')
    scheduler.add_job(run_trading, 'interval', minutes=1, id='trading_job')
    scheduler.add_job(optimize_parameters, 'interval', minutes=30, id='optimization_job')
    scheduler.add_job(retraining_job, 'interval', minutes=30, id='retraining_job')

    def shutdown(signum, frame):
        logger.info(f"Signal {signum} reçu, arrêt du scheduler.")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Planificateur démarré. Les tâches d'automatisation s'exécutent selon le planning.")
    scheduler.start()

if __name__ == '__main__':
    # Redémarrage automatique du scheduler en cas de crash non prévu
    while True:
        try:
            start_scheduler()
        except Exception as e:
            logger.error(f"Erreur inattendue dans le scheduler: {e}. Redémarrage dans 5 secondes...")
            time.sleep(5)
