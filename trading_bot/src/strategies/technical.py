import pandas as pd
import numpy as np
import time
import logging
import tensorflow as tf
import os
import sys
from tensorflow.keras.models import load_model
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.config_features import LSTM_FEATURES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def extrapolate_dataframe(df, target_length, window=2):
    current_length = len(df)
    pad_rows = target_length - current_length
    if pad_rows <= 0:
        return df.copy()
    last_row = df.iloc[-1]
    if current_length >= window + 1:
        differences = []
        for i in range(1, window+1):
            diff = df.iloc[-i] - df.iloc[-i-1]
            differences.append(diff)
        slope = sum(differences) / window
    else:
        slope = pd.Series(0, index=last_row.index)
    new_rows = []
    for i in range(pad_rows):
        new_row = last_row + slope * (i + 1)
        new_rows.append(new_row)
    padding_df = pd.DataFrame(new_rows)
    result_df = pd.concat([df, padding_df], ignore_index=True)
    return result_df

class TechnicalStrategy:
    def __init__(self):
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\trading_bot\\src\\models",
                                      "LSTM_trading_model_final_pre-entrainement.keras")
            scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\trading_bot\\src\\models",
                                       "scaler_pre-entrainement.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.lstm_model = load_model(model_path,
                custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal},
                compile=False)
            self.scaler = joblib.load(scaler_path)
            print("✅ LSTM Model et Scaler chargés avec succès")
        except Exception as e:
            print(f"❌ Échec du chargement du modèle ou du scaler : {e}")
            self.lstm_model = None
            self.scaler = None

    def compute_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, df, short_window=12, long_window=26, signal_window=9):
        ema_short = df['close'].ewm(span=short_window, adjust=False).mean()
        ema_long = df['close'].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    def compute_bollinger_bands(self, df, window=20, num_std=2):
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def compute_ichimoku(self, df, conversion_period=9, base_period=26, leading_period=52, displacement=26):
        if len(df) < leading_period:
            logger.error("🚨 Pas assez de données pour calculer l'Ichimoku.")
            return (pd.Series(dtype=float), pd.Series(dtype=float),
                    pd.Series(dtype=float), pd.Series(dtype=float))
        conversion_line = (df['high'].rolling(window=conversion_period).max() +
                           df['low'].rolling(window=conversion_period).min()) / 2
        base_line = (df['high'].rolling(window=base_period).max() +
                     df['low'].rolling(window=base_period).min()) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        leading_span_b = ((df['high'].rolling(window=leading_period).max() +
                           df['low'].rolling(window=leading_period).min()) / 2).shift(displacement)
        return conversion_line, base_line, leading_span_a, leading_span_b

    def compute_adx(self, df, period=14):
        if not {'high', 'low', 'close'}.issubset(df.columns):
            raise ValueError("Les colonnes 'high','low','close' sont requises pour l'ADX.")
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low,
                        abs(high - close.shift(1)),
                        abs(low - close.shift(1))], axis=1).max(axis=1)
        up_move = high.diff()
        down_move = low.diff().abs()
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr_smooth = tr.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
        pos_dm_smooth = pos_dm.rolling(window=period, min_periods=period).sum()
        neg_dm_smooth = neg_dm.rolling(window=period, min_periods=period).sum()
        pos_di = 100 * (pos_dm_smooth / (tr_smooth + 1e-10))
        neg_di = 100 * (neg_dm_smooth / (tr_smooth + 1e-10))
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)).replace([np.inf, -np.inf], np.nan)
        adx = dx.rolling(window=period, min_periods=period).mean()
        return adx

    def compute_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def compute_ema(self, df, span=20):
        return df['close'].ewm(span=span, adjust=False).mean()
    
    def predict_lstm(self, df, sequence_length=40):
        """
        Prédiction des prix futurs à partir du modèle LSTM.
        Prépare les données en calculant les indicateurs supplémentaires, en comblant les NaN
        par interpolation linéaire puis ffill/bfill, et en s'assurant que la séquence a au moins sequence_length lignes.
        """
        if self.lstm_model is None:
            logger.error("❌ LSTM Model non chargé. Skip prediction.")
            return None

        df = df.copy()

        # Calcul des indicateurs supplémentaires
        df.loc[:, "ADX"] = self.compute_adx(df)
        df.loc[:, "ATR"] = self.compute_atr(df)
        df.loc[:, "SMA_20"] = df["close"].rolling(window=20).mean()
        df.loc[:, "SMA_50"] = df["close"].rolling(window=50).mean()
        df.loc[:, "EMA_20"] = self.compute_ema(df, span=20)
        upper_band, lower_band = self.compute_bollinger_bands(df)
        df.loc[:, "Bollinger_High"] = upper_band
        df.loc[:, "Bollinger_Low"] = lower_band
        df.loc[:, "RSI"] = self.compute_rsi(df)
        macd, macd_signal = self.compute_macd(df)
        df.loc[:, "MACD"] = macd
        df.loc[:, "MACD_signal"] = macd_signal

        # Vérifier et loguer les colonnes disponibles
        required_cols = LSTM_FEATURES
        logger.info(f"Colonnes disponibles pour LSTM: {list(df.columns)}")
        logger.info(f"Colonnes attendues (LSTM_FEATURES): {required_cols}")
        
        df_required = df[required_cols].copy()
        
        # Traitement des NaN dans df_required
        df_required.interpolate(method='linear', limit_direction='both', inplace=True)
        df_required.ffill(inplace=True)
        df_required.bfill(inplace=True)
        
        # Vérifier la taille de la séquence après traitement
        if len(df_required) < sequence_length:
            df_required = extrapolate_dataframe(df_required, sequence_length, window=2)
            logger.warning("Séquence incomplète pour LSTM. Extrapolation appliquée pour atteindre la taille minimale.")

        if df_required.isnull().any().any():
            logger.error("Des NaN persistent dans df_required après traitement.")
            return None

        logger.info(f"Dimension du DataFrame utilisé pour LSTM après traitement: {df_required.shape}")

        df_scaled = self.scaler.transform(df_required)
        sequences = []
        for i in range(len(df_scaled) - sequence_length + 1):
            sequences.append(df_scaled[i:i+sequence_length])
        if not sequences:
            logger.error("🚨 Aucune séquence générée pour la prédiction LSTM.")
            return None
        X = np.array(sequences)
        logger.info(f"📊 Dimension de X pour LSTM: {X.shape}")
        predictions = self.lstm_model.predict(X, batch_size=32)
        if predictions is None or len(predictions) == 0:
            logger.error("🚨 Erreur lors de la prédiction LSTM.")
            return None
        lstm_prediction = predictions[-1][0]
        logger.info(f"🔮 Dernière prédiction LSTM: {lstm_prediction}")
        return lstm_prediction
