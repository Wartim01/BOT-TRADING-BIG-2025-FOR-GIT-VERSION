import os
import sys
import logging
import time
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Ajouter le dossier racine du projet dans le path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import de la configuration centralisée des features (si nécessaire)
from utils.config_features import LSTM_FEATURES

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
    # Chemin vers le fichier d'historique de pré-entrainement
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    input_file = os.path.join(base_data_path, "historique_pre-entrainement.csv")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {input_file} : {e}")
        return

    # Vérifier la présence des colonnes attendues
    required_columns = LSTM_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"🚨 Colonnes manquantes dans '{input_file}' : {missing_columns}")

    df.dropna(inplace=True)
    logger.info(f"Nombre de lignes après suppression des NaN: {len(df)}")

    # Sélection des features utilisées pour le modèle
    features_used = LSTM_FEATURES
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features_used])
    # Sauvegarder le scaler avec un suffixe pour pré-entrainement
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/scaler_pre-entrainement.pkl"))
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Scaler pré-entrainement sauvegardé avec {len(df)} lignes utilisées.")

    # Préparation des séquences pour le modèle LSTM
    SEQ_LEN = 60  # Nombre de bougies par séquence

    def create_sequences(data, seq_len, augment=True):
        X, y = [], []
        for i in range(len(data) - seq_len):
            seq = data[i:i+seq_len].copy()
            if augment:
                seq += np.random.normal(0, 0.002, seq.shape)
            X.append(seq)
            y.append(data[i+seq_len, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, SEQ_LEN, augment=True)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    logger.info(f"📊 {X_train.shape[0]} entrées pour l'entraînement, {X_val.shape[0]} pour la validation.")

    # Définition du modèle LSTM optimisé
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
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
    lr_schedule = CosineDecay(initial_lr, decay_steps=1000, alpha=0.1)
    optimizer = AdamW(learning_rate=lr_schedule)
    from tensorflow.keras.losses import Huber
    model.compile(optimizer=optimizer, loss=Huber(), metrics=["mae"])

    # Définir les chemins pour sauvegarder le checkpoint et le modèle final (version pré-entrainement)
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/LSTM_checkpoint_pre-entrainement.weights.h5"))
    final_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/LSTM_trading_model_final_pre-entrainement.keras"))

    early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    logger.info("🚀 Début de l'entraînement pré-entrainement...")
    try:
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            logger.info("✅ Poids du modèle pré-entrainement restaurés avec succès ! Reprise de l'entraînement...")
    except Exception as e:
        logger.info("⚠️ Aucun checkpoint pré-entrainement trouvé ou erreur lors du chargement, entraînement depuis zéro.")

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=128,
            callbacks=[early_stopping, checkpoint_callback],
            verbose=2
        )
    except KeyboardInterrupt:
        logger.warning("Interruption détectée ! Sauvegarde du checkpoint pré-entrainement avant de quitter...")
        model.save_weights(checkpoint_path)
        logger.info("✅ Checkpoint pré-entrainement sauvegardé. Arrêt de l'entraînement.")
        return
    finally:
        model.save(final_model_path)
        logger.info(f"✅ Modèle final pré-entrainement sauvegardé sous '{final_model_path}'.")

    # Optionnel : Test avec TTA (Test-Time Augmentation)
    def predict_with_tta(model, X_sample, n_augment=10):
        predictions = []
        for _ in range(n_augment):
            noise = np.random.normal(0, 0.002, X_sample.shape)
            pred = model.predict(X_sample + noise)
            predictions.append(pred)
        return np.mean(predictions, axis=0)

    logger.info("🧠 Test-Time Augmentation (TTA) en cours...")
    X_test_sample = X_val[:10]
    tta_predictions = predict_with_tta(model, X_test_sample)
    logger.info(f"📈 Prédictions finales améliorées : {tta_predictions}")

if __name__ == "__main__":
    main()

def compute_composite_score_with_lstm(strategy, df_slice, weights):
    # ...existing code computing technical indicators...
    tech_score = (
        rsi_signal * weights.get('rsi', 0.20) +
        macd_signal * weights.get('macd', 0.20) +
        boll_signal * weights.get('bollinger', 0.15) +
        ichimoku_signal * weights.get('ichimoku', 0.15) +
        adx_signal * weights.get('adx', 0.10) +
        vol_signal * weights.get('volume', 0.10) +
        ma_signal * weights.get('ma_crossover', 0.10)
    )
    
    # Ensure the DataFrame passed to LSTM prediction has the required features.
    for col in ["high", "low"]:
        if col not in df_slice.columns:
            df_slice[col] = df_slice["close"]
    
    # Signal LSTM - catch any prediction errors and fallback to current close price
    try:
        predicted_price = strategy.predict_lstm(df_slice)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction LSTM : {e}. Utilisation de la valeur actuelle.")
        predicted_price = last_close  # fallback: use current close price
    lstm_ratio = (predicted_price - last_close) / max(last_close, 1e-6)
    lstm_ratio = max(-1, min(1, lstm_ratio))
    lstm_weight = weights.get('lstm', 0.5)
    final_score = tech_score + lstm_weight * lstm_ratio
    return final_score
