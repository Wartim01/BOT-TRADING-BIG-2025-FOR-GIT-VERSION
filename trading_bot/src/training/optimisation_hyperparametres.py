import numpy as np
import tensorflow as tf
import random
import os
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from hyperopt import fmin, tpe, hp, Trials
import joblib
from tqdm import tqdm

# Fixer la graine pour reproductibilité
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Chargement des données prétraitées depuis les fichiers NumPy
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Définition de l'espace de recherche pour Hyperopt
space = {
    "lstm_units": hp.choice("lstm_units", [64, 128, 256, 512]),
    "gru_units": hp.choice("gru_units", [64, 128, 256]),
    "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
    "batch_size": hp.choice("batch_size", [32, 64, 128]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
    "epochs": hp.choice("epochs", [50, 100, 200])
}

# Chargement des essais Hyperopt précédents s'ils existent
if os.path.exists("hyperopt_trials.pkl"):
    trials = joblib.load("hyperopt_trials.pkl")
    logger.info("Chargement des essais Hyperopt précédents.")
else:
    trials = Trials()
    logger.info("Nouveau run Hyperopt.")

# Chargement des meilleurs hyperparamètres si disponibles
if os.path.exists("best_hyperparameters.pkl"):
    best_params = joblib.load("best_hyperparameters.pkl")
    logger.info(f"Hyperparamètres optimaux rechargés: {best_params}")
else:
    best_params = None

best_loss = best_params['loss'] if best_params and 'loss' in best_params else float('inf')

# Wrapper pour random state compatible avec Hyperopt
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
    
    model = Sequential([
        LSTM(params["lstm_units"], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(params["dropout_rate"]),
        BatchNormalization(),
        GRU(params["gru_units"], return_sequences=False),
        Dropout(params["dropout_rate"]),
        Dense(50, activation="relu"),
        Dense(1, activation="linear")
    ])
    
    optimizer = Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss="mae")
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(
        filepath="model_checkpoint.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    if os.path.exists("model_checkpoint.weights.h5"):
        try:
            model.load_weights("model_checkpoint.weights.h5")
            logger.info("Poids chargés depuis le checkpoint.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du checkpoint: {e}")
    
    with tqdm(total=params["epochs"], desc="Entraînement du modèle", unit="epoch") as pbar:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=[early_stopping, checkpoint_callback,
                       LambdaCallback(on_epoch_end=lambda epoch, logs: pbar.update(1))]
        )
    
    val_loss = min(history.history["val_loss"])
    logger.info(f"Validation Loss: {val_loss:.5f}")
    return {"loss": val_loss, "status": "ok", "params": params}

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
        logger.info(f"Après {current_evals} essais, meilleur loss = {current_best_loss}")
        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_params = trials.best_trial['result']['params']
            joblib.dump(trials, "hyperopt_trials.pkl")
            joblib.dump({**best_params, "loss": best_loss}, "best_hyperparameters.pkl")
            logger.info(f"Nouveau meilleur modèle trouvé: {best_params} avec loss = {best_loss}")
    except KeyboardInterrupt:
        logger.warning("Interruption détectée. Sauvegarde des essais actuels...")
        joblib.dump(trials, "hyperopt_trials.pkl")
        if best_params is not None:
            joblib.dump({**best_params, "loss": best_loss}, "best_hyperparameters.pkl")
        break

logger.info("Optimisation terminée et sauvegardée !")
