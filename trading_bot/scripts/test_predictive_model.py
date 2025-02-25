import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import asyncio  # Pour gérer la coroutine éventuelle
import json
import logging
import time
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Import du module central des features
from src.utils.config_features import LSTM_FEATURES

# Modules internes
from src.data.collector import DataCollector
from src.data.processor import DataProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config():
    """
    Charge la configuration depuis src/config/config.json
    pour récupérer l'API key et l'API secret.
    """
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/config/config.json'))
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("Configuration loaded")
    return config

def main():
    # ==================== 1) CHARGEMENT CONFIG & DONNÉES ====================
    config = load_config()
    
    # Récupérer les clés API selon le mode ("production" ou "testnet")
    mode = config["mode"]
    api_key = config["binance"][mode]["API_KEY"]
    api_secret = config["binance"][mode]["API_SECRET"]

    collector = DataCollector(api_key, api_secret)
    
    # Utiliser asyncio.run() pour exécuter la coroutine get_historical_data
    data = asyncio.run(collector.get_historical_data('BTCUSDT', '1h', '3 month ago UTC'))
    if data is None:
        logger.error("Failed to fetch historical data")
        return

    processor = DataProcessor()
    df = processor.process_data(data)
    logger.info(f"Dataframe rows after basic processing: {len(df)}")

    # ==================== 2) CHARGEMENT DU MODELE & SCALER ====================
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/models/LSTM_trading_model_final.keras'))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/models/scaler.pkl'))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle final non trouvé : {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler final non trouvé : {scaler_path}")

    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    # ==================== 3) PRÉPARATION IDENTIQUE A L'ENTRAÎNEMENT ====================
    # Utiliser la configuration centralisée des features
    required_cols = LSTM_FEATURES
    
    # Si certaines colonnes ne sont pas présentes, on calcule les indicateurs
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        import ta
        # SMA et EMA
        if "SMA_20" in missing:
            df["SMA_20"] = df["close"].rolling(window=20).mean()
        if "SMA_50" in missing:
            df["SMA_50"] = df["close"].rolling(window=50).mean()
        if "EMA_20" in missing:
            df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
        # RSI
        if "RSI" in missing:
            df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        # MACD et son signal
        if "MACD" in missing or "MACD_signal" in missing:
            macd = ta.trend.MACD(df["close"])
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()
        # Bollinger Bands
        if "Bollinger_High" in missing or "Bollinger_Low" in missing:
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["Bollinger_High"] = bb.bollinger_hband()
            df["Bollinger_Low"] = bb.bollinger_lband()
        # ATR
        if "ATR" in missing:
            df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        # ADX
        if "ADX" in missing:
            df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        df.dropna(inplace=True)
    
    # Vérification des colonnes après calcul
    missing_after = [c for c in required_cols if c not in df.columns]
    if missing_after:
        logger.error(f"Missing columns for LSTM input after processing: {missing_after}")
        return

    data_scaled = scaler.transform(df[required_cols])

    # ==================== 4) CREATION DES SEQUENCES ====================
    seq_len = 60  # Selon votre entraînement
    def create_sequences(array, seq_len=60):
        X = []
        for i in range(len(array) - seq_len):
            X.append(array[i:i+seq_len])
        return np.array(X)

    X_full = create_sequences(data_scaled, seq_len=seq_len)
    logger.info(f"X_full shape for predictions: {X_full.shape}")
    if len(X_full) == 0:
        logger.error("Not enough data to create sequences. Aborting.")
        return

    # ==================== 5) PREDICTIONS & COMPARAISONS ====================
    predictions = model.predict(X_full)
    logger.info(f"Predictions shape: {predictions.shape}")

    dummy = np.zeros((predictions.shape[0], len(required_cols)))
    dummy[:,0] = predictions.flatten()
    inv_predictions = scaler.inverse_transform(dummy)[:,0]

    actuals = df["close"].values[seq_len:]
    logger.info("First 10 predictions vs actuals:")
    limit = min(10, len(inv_predictions))
    for i in range(limit):
        logger.info(f"Pred {i+1} = {inv_predictions[i]:.2f} | Real = {actuals[i]:.2f}")

    logger.info("Last 10 predictions vs actuals:")
    for i in range(-10, 0):
        if len(inv_predictions) + i < 0:
            continue
        idx = len(inv_predictions) + i
        logger.info(f"Pred {idx+1} = {inv_predictions[i]:.2f} | Real = {actuals[i]:.2f}")

    logger.info("Test predictive model terminé.")

if __name__ == '__main__':
    main()
