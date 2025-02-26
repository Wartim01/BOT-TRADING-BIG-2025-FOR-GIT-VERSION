import os
import sys
import json
import logging
import joblib
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from tqdm import tqdm

# Fixer la graine pour reproductibilité
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration (supposé se trouver dans trading_bot/config)
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/config.json"))

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration mise à jour.")

# IMPORTANT : Mettre à jour le chemin pour importer le module backtest.
# Ici, nous supposons que le backtest se trouve dans trading_bot/scripts/backtest.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.backtest import run_backtest

# Updated objective_function signature:
def objective_function(lstm_units, gru_units, dropout_rate, batch_size, learning_rate, epochs, 
                         buy_threshold, sell_threshold, max_risk_per_trade, stop_loss):
    config = load_config()
    # Mise à jour des paramètres de trading et gestion du risque
    config["trading"]["buy_threshold"] = buy_threshold
    config["trading"]["sell_threshold"] = sell_threshold
    config["trading"]["risk_management"] = config["trading"].get("risk_management", {})
    config["trading"]["risk_management"]["max_risk_per_trade"] = max_risk_per_trade
    config["trading"]["risk_management"]["stop_loss"] = stop_loss
    save_config(config)
    try:
        results = run_backtest(config)
    except Exception as e:
        logger.error(f"Erreur lors du backtest: {e}")
        return -1e6
    if not results:
        return -1e6
    pnl = results.get("pnl", 0)
    win_rate = results.get("win_rate", 0)
    max_drawdown = results.get("max_drawdown", 1)
    score = pnl * (win_rate / 100) - max_drawdown
    logger.info(f"Score: {score:.2f} (PnL: {pnl:.2f}, Win Rate: {win_rate:.2f}%, Drawdown: {max_drawdown:.2f}%)")
    return score

# Espace de recherche incluant les seuils et les paramètres de gestion des risques
space = {
    "lstm_units": hp.choice("lstm_units", [64, 128, 256, 512]),
    "gru_units": hp.choice("gru_units", [64, 128, 256]),
    "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
    "epochs": hp.choice("epochs", [50, 100, 200]),
    "buy_threshold": hp.uniform("buy_threshold", 0.02, 0.1),
    "sell_threshold": hp.uniform("sell_threshold", -0.1, -0.02),
    "max_risk_per_trade": hp.uniform("max_risk_per_trade", 0.005, 0.02),
    "stop_loss": hp.uniform("stop_loss", 0.01, 0.05)
    }

# Correction du chemin pour sauvegarder les essais Hyperopt dans le dossier config
trials_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/hyperopt_trials.pkl"))
if os.path.exists(trials_path):
    trials = joblib.load(trials_path)
    logger.info("Chargement des essais Hyperopt précédents.")
else:
    trials = Trials()
    logger.info("Nouvelle instance d'essais Hyperopt.")

if os.path.exists("best_hyperparameters.pkl"):
    best_params = joblib.load("best_hyperparameters.pkl")
    logger.info(f"Hyperparamètres optimaux rechargés: {best_params}")
else:
    best_params = None

best_loss = best_params['loss'] if best_params and 'loss' in best_params else float('inf')

class RandomStateWrapper:
    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)
    def integers(self, low, high=None, size=None, dtype=int, endpoint=False):
        return self.rs.randint(low, high=high, size=size, dtype=dtype)
    def __getattr__(self, attr):
        return getattr(self.rs, attr)

rstate = RandomStateWrapper(seed_value)

def train_and_evaluate(params):
    logger.info(f"Test des paramètres: {params}")
    score = objective_function(**params)
    # Nous cherchons à maximiser le score, donc nous minimisons -score
    return {"loss": -score, "status": "ok", "params": params}

max_evals = 20
current_evals = len(trials.trials)

while current_evals < max_evals:
    try:
        best = fmin(
            fn=train_and_evaluate,
            space=space,
            algo=tpe.suggest,
            max_evals=current_evals + 1,
            trials=trials,
            rstate=rstate
        )
        current_evals = len(trials.trials)
        current_best_loss = trials.best_trial['result']['loss']
        logger.info(f"Après {current_evals} essais, meilleur score = {-current_best_loss}")
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_params = trials.best_trial['result']['params']
            joblib.dump(trials, trials_path)
            joblib.dump({**best_params, "loss": best_loss}, "best_hyperparameters.pkl")
            logger.info(f"Nouveaux hyperparamètres optimaux: {best_params} avec score = {-best_loss}")
    except KeyboardInterrupt:
        logger.warning("Interruption détectée. Sauvegarde des essais actuels...")
        joblib.dump(trials, trials_path)
        if best_params is not None:
            joblib.dump({**best_params, "loss": best_loss}, "best_hyperparameters.pkl")
        break

logger.info("Optimisation terminée et sauvegardée !")
