import os
import sys
import pandas as pd
import numpy as np
import joblib
import ta
import logging
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Ajouter le dossier racine du projet dans le path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import de la configuration centralisée des features
from utils.config_features import LSTM_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime.s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Chemin absolu du fichier historique utilisé pour le pré‑entraînement
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/historique_pre-entrainement.csv"))
    logger.info(f"Chargement des données depuis {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {input_file} : {e}")
        return

    # Vérifier que les colonnes essentielles existent
    required_columns = ["close", "high", "low", "volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"🚨 Colonnes manquantes dans '{input_file}' : {missing_cols}")

    # Ajout des indicateurs techniques
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["Bollinger_High"] = bb.bollinger_hband()
    df["Bollinger_Low"] = bb.bollinger_lband()
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

    initial_len = len(df)
    df.dropna(inplace=True)
    logger.info(f"Suppression des NaN : {initial_len - len(df)} lignes retirées.")

    # Utiliser la liste centralisée des features pour le modèle LSTM
    features_used = LSTM_FEATURES
    scaler = MinMaxScaler()
    scaler.fit(df[features_used])
    # Sauvegarder le scaler dans le dossier models (correction de chemin)
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/scaler_pre-entrainement.pkl"))
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Scaler sauvegardé sous '{scaler_path}' ! Entraîné avec {len(df)} lignes de données.")

    # Mise à jour (optionnelle) du fichier historique enrichi pour le pré‑entraînement
    df.to_csv(input_file, index=False)
    logger.info(f"✅ Fichier '{input_file}' mis à jour avec les indicateurs techniques.")

    # Préparation des séquences pour le modèle LSTM
    SEQ_LEN = 60  # Nombre de bougies par séquence

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in tqdm(range(len(data) - seq_len), desc="Création des séquences", unit="seq"):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 0])  # On prédit la valeur de "close"
        return np.array(X), np.array(y)

    scaled_data = scaler.transform(df[features_used])
    X, y = create_sequences(scaled_data, SEQ_LEN)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("🚨 Problème lors de la création des séquences, vérifiez la taille des données.")

    # Séparation en ensembles d'entraînement et de validation (80/20)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    logger.info(f"✅ Données préparées : X_train={X_train.shape}, X_val={X_val.shape}")

    # Sauvegarder les données prétraitées dans le dossier data avec un suffixe spécifique
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    np.save(os.path.join(base_data_path, "X_train_pre-entrainement.npy"), X_train)
    np.save(os.path.join(base_data_path, "y_train_pre-entrainement.npy"), y_train)
    np.save(os.path.join(base_data_path, "X_val_pre-entrainement.npy"), X_val)
    np.save(os.path.join(base_data_path, "y_val_pre-entrainement.npy"), y_val)
    logger.info("✅ Fichiers NumPy pré‑entrainement sauvegardés dans le dossier data.")

if __name__ == "__main__":
    main()
