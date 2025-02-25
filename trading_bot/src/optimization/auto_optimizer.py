import json
import logging
import os
import sys
from bayes_opt import BayesianOptimization  # Ensure installation with: pip install bayesian-optimization

# Ajouter le dossier racine (BOT TRADING BIG 2025) à sys.path pour pouvoir importer le package trading_bot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from trading_bot.scripts.backtest import run_backtest  # Import de la fonction de backtest existante

# Définir CONFIG_PATH en pointant vers le fichier de configuration
CONFIG_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../trading_bot/config')), "config.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration mise à jour.")

def objective_function(rsi_weight, macd_weight, bollinger_weight, ichimoku_weight, adx_weight, volume_weight, lstm_weight, buy_threshold, sell_threshold):
    """
    Fonction objectif qui ajuste les poids des signaux et les seuils d'entrée/sortie.
    Cette fonction met à jour la configuration, lance un backtest et retourne un score basé sur les performances.
    """
    config = load_config()
    config["signal_weights"]["rsi"] = rsi_weight
    config["signal_weights"]["macd"] = macd_weight
    config["signal_weights"]["bollinger"] = bollinger_weight
    config["signal_weights"]["ichimoku"] = ichimoku_weight
    config["signal_weights"]["adx"] = adx_weight
    config["signal_weights"]["volume"] = volume_weight
    config["signal_weights"]["lstm"] = lstm_weight
    config["trading"]["buy_threshold"] = buy_threshold
    config["trading"]["sell_threshold"] = sell_threshold

    save_config(config)

    results = run_backtest(config)
    if not results:
        return -1e6

    pnl = results.get("pnl", 0)
    win_rate = results.get("win_rate", 0)
    max_drawdown = results.get("max_drawdown", 1)
    score = pnl * (win_rate / 100) - max_drawdown
    logger.info(f"Score: {score:.2f} (PnL: {pnl:.2f}, Win Rate: {win_rate:.2f}%, Drawdown: {max_drawdown:.2f}%)")
    return score

def optimize_parameters():
    pbounds = {
        "rsi_weight": (0.1, 0.3),
        "macd_weight": (0.1, 0.3),
        "bollinger_weight": (0.1, 0.2),
        "ichimoku_weight": (0.1, 0.2),
        "adx_weight": (0.05, 0.15),
        "volume_weight": (0.05, 0.15),
        "lstm_weight": (0.3, 0.7),
        "buy_threshold": (0.4, 0.6),
        "sell_threshold": (-0.6, -0.4)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )
    
    optimizer.maximize(init_points=5, n_iter=10)
    
    best_params = optimizer.max["params"]
    logger.info("Meilleure configuration trouvée:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    config = load_config()
    config["signal_weights"]["rsi"] = best_params["rsi_weight"]
    config["signal_weights"]["macd"] = best_params["macd_weight"]
    config["signal_weights"]["bollinger"] = best_params["bollinger_weight"]
    config["signal_weights"]["ichimoku"] = best_params["ichimoku_weight"]
    config["signal_weights"]["adx"] = best_params["adx_weight"]
    config["signal_weights"]["volume"] = best_params["volume_weight"]
    config["signal_weights"]["lstm"] = best_params["lstm_weight"]
    config["trading"]["buy_threshold"] = best_params["buy_threshold"]
    config["trading"]["sell_threshold"] = best_params["sell_threshold"]
    
    save_config(config)
    logger.info("Optimisation terminée et configuration mise à jour.")

if __name__ == "__main__":
    optimize_parameters()
