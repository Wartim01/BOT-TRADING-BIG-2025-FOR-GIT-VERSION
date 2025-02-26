import os
import sys
import logging
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
GRU = keras.layers.GRU
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
LayerNormalization = keras.layers.LayerNormalization
Adam = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

# Ajouter le dossier racine du projet dans le path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import de la configuration centralisée des features
from src.utils.config_features import LSTM_FEATURES

# Configuration de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixer la graine pour la reproductibilité
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def main():
    # Chemins absolus vers les fichiers de données prétraitées
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/data"))
    X_train_path = os.path.join(base_data_path, "X_train.npy")
    y_train_path = os.path.join(base_data_path, "y_train.npy")
    X_val_path   = os.path.join(base_data_path, "X_val.npy")
    y_val_path   = os.path.join(base_data_path, "y_val.npy")
    
    # Chargement des données
    try:
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_val   = np.load(X_val_path)
        y_val   = np.load(y_val_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données NumPy: {e}")
        return

    logger.info(f"✅ Données chargées : X_train={X_train.shape}, X_val={X_val.shape}")

    # Vérifier que le nombre de features correspond à la configuration centralisée
    if X_train.shape[2] != len(LSTM_FEATURES):
        logger.error(f"Incohérence des features : X_train a {X_train.shape[2]} features, mais LSTM_FEATURES contient {len(LSTM_FEATURES)}")
        raise ValueError("Incohérence des features entre les données d'entraînement et la configuration centrale")
    
    # Définition du modèle LSTM optimisé
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.4),
        BatchNormalization(),
        
        GRU(128, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        LayerNormalization(),
        
        Dense(50, activation="relu"),
        Dense(25, activation="relu"),
        Dense(1, activation="linear")
    ])

    initial_lr = 0.001
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss="mae")
    logger.info("Modèle compilé.")

    # Chemin pour sauvegarder le checkpoint et le modèle final
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/models/model_checkpoint.weights.h5"))
    final_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/models/LSTM_trading_model_final.keras"))

    # Callback pour early stopping et checkpointing
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Charger les poids depuis le checkpoint s'il existe
    if os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
            logger.info("🔄 Poids chargés depuis le checkpoint, reprise de l'entraînement...")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du checkpoint: {e}")

    # Entraînement du modèle
    epochs = 100  # Vous pouvez ajuster ce nombre selon vos besoins
    batch_size = 128
    logger.info("🚀 Début de l'entraînement...")
    logger.info("🚀 Début de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint_callback],
        verbose=1
    )
    # Définition d'un scaler identité pour l'inversion de la normalisation
    class IdentityScaler:
        def inverse_transform(self, data):
            return data
    target_scaler = IdentityScaler()

    # Évaluation sur l'ensemble de validation
    preds = model.predict(X_val)
    # Inversion de la normalisation
    preds_inversed = target_scaler.inverse_transform(preds)
    y_val_inversed = target_scaler.inverse_transform(y_val.reshape(-1,1))
    preds = model.predict(X_val)
    # Inversion de la normalisation
    preds_inversed = target_scaler.inverse_transform(preds)
    y_val_inversed = target_scaler.inverse_transform(y_val.reshape(-1,1))

    mae = np.mean(np.abs(y_val_inversed - preds_inversed))
    rmse = np.sqrt(np.mean((y_val_inversed - preds_inversed)**2))
    logger.info(f"MAE sur validation: {mae:.5f}")
    logger.info(f"RMSE sur validation: {rmse:.5f}")

    # Sauvegarde du modèle final
    model.save(final_model_path)
    logger.info(f"✅ Modèle final sauvegardé sous '{final_model_path}'.")
if __name__ == "__main__":
    main()
