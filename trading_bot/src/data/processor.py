import numpy as np
import pandas as pd
import time
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Liste des colonnes attendues pour le traitement
REQUIRED_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]

class DataProcessor:
    """
    Classe pour traiter les données brutes (par exemple, les klines récupérées depuis Binance)
    et les transformer en un DataFrame utilisable par les modèles de prédiction.
    """
    def validate_columns(self, df):
        """Vérifie que toutes les colonnes requises sont présentes."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.error(f"Colonnes manquantes dans les données : {missing}")
            raise ValueError(f"Les données brutes ne contiennent pas les colonnes suivantes : {missing}")
        return True

    def process_data(self, raw_data):
        """
        Transforme les données brutes en DataFrame :
          - Création du DataFrame avec les colonnes attendues
          - Conversion des colonnes numériques en float
          - Conversion des timestamps
          - Traitement des NaN par interpolation linéaire, ffill et bfill
          - Suppression des rares lignes restantes avec des NaN
          - Affichage de la forme finale et des statistiques NaN
        """
        df = pd.DataFrame(raw_data, columns=REQUIRED_COLUMNS)
        self.validate_columns(df)
    
        numeric_cols = ["open", "high", "low", "close", "volume", 
                        "quote_asset_volume", "num_trades", 
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        try:
            df[numeric_cols] = df[numeric_cols].astype(float)
        except Exception as e:
            logger.error(f"Erreur lors de la conversion des colonnes numériques : {e}")
            raise

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        except Exception as e:
            logger.error(f"Erreur lors de la conversion des timestamps : {e}")
            raise

        # Statistiques initiales des NaN
        logger.info("Statistiques NaN initiales:")
        logger.info(df.isnull().sum())

        # Traitement des NaN : interpolation linéaire, puis ffill et bfill
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    
        # Vérifier et supprimer les éventuelles lignes restantes avec des NaN
        if df.isnull().any().any():
            before = df.shape[0]
            df.dropna(inplace=True)
            after = df.shape[0]
            logger.warning(f"Des NaN persistaient et ont été supprimés. Lignes supprimées: {before - after}")

        # Afficher la forme finale et les statistiques NaN finales
        logger.info(f"DataFrame final: {df.shape}")
        logger.info("Statistiques NaN finales:")
        logger.info(df.isnull().sum())
    
        return df



class QuantitativeStrategy:
    def compute_trend(self, df, window=30):
        start_time = time.time()
        df = df.copy()
        df['trend'] = np.nan
        for i in range(window, len(df)):
            X = np.arange(window).reshape(-1, 1)
            y = df['close'].iloc[i-window:i].values
            model = LinearRegression()
            model.fit(X, y)
            df.loc[df.index[i], 'trend'] = model.coef_[0]
        duration = time.time() - start_time
        logger.info(f"compute_trend executed in {duration:.4f} seconds")
        return df

    def compute_zscore(self, series):
        start_time = time.time()
        zscore = (series - series.mean()) / series.std()
        duration = time.time() - start_time
        logger.info(f"compute_zscore executed in {duration:.4f} seconds")
        return zscore

class ComplexPredictiveModel:
    def __init__(self, sequence_length=60, epochs=150, batch_size=32, features=None, target_feature='close'):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        if features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']
        else:
            self.features = features
        self.target_feature = target_feature
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.bias = 0.0

    def create_sequences(self, features_scaled, target_scaled):
        sequences = []
        targets = []
        for i in range(len(features_scaled) - self.sequence_length):
            seq = features_scaled[i:i+self.sequence_length]
            target = target_scaled[i+self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, df):
        start_time = time.time()
        # Extraction des caractéristiques et de la cible
        data_features = df[self.features].values
        data_target = df[[self.target_feature]].values
        # Appliquer la normalisation
        features_scaled = self.feature_scaler.fit_transform(data_features)
        target_scaled = self.target_scaler.fit_transform(data_target)
        X, y = self.create_sequences(features_scaled, target_scaled)
        X = X.reshape(X.shape[0], X.shape[1], len(self.features))
        self.build_model((X.shape[1], len(self.features)))
        early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-5, verbose=1)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2,
                       verbose=1, callbacks=[early_stop, reduce_lr])
        duration = time.time() - start_time
        logger.info(f"Model trained in {duration:.4f} seconds")
        self.calibrate(df)

    def calibrate(self, df):
        data_features = df[self.features].values
        data_target = df[[self.target_feature]].values
        features_scaled = self.feature_scaler.transform(data_features)
        target_scaled = self.target_scaler.transform(data_target)
        X, y = self.create_sequences(features_scaled, target_scaled)
        X = X.reshape(X.shape[0], X.shape[1], len(self.features))
        predictions = self.model.predict(X, verbose=0)
        predictions_inversed = self.target_scaler.inverse_transform(predictions)
        targets_inversed = self.target_scaler.inverse_transform(y)
        self.bias = np.mean(targets_inversed - predictions_inversed)
        logger.info(f"Calibration bias computed: {self.bias}")

    def predict(self, df):
        """
        Effectue une prédiction sur un DataFrame donné. Pour s'assurer que
        les colonnes utilisées correspondent exactement aux features utilisées lors de l'entraînement,
        on vérifie si le scaler dispose d'un attribut `feature_names_in_` et on réordonne le DataFrame en conséquence.
        """
        df_copy = df.copy()
        # Si le scaler possède 'feature_names_in_', utilisez-les pour filtrer et réordonner le DataFrame.
        if hasattr(self.feature_scaler, "feature_names_in_"):
            required_features = list(self.feature_scaler.feature_names_in_)
            missing = [col for col in required_features if col not in df_copy.columns]
            if missing:
                logger.error(f"Les features suivantes sont manquantes dans les données : {missing}")
                raise ValueError(f"Missing features: {missing}")
            df_copy = df_copy[required_features]
        else:
            df_copy = df_copy[self.features]
        
        data_features = df_copy.values
        # Pour la cible, si elle est présente dans df_copy on l'utilise, sinon on la récupère à partir de self.target_feature
        if self.target_feature in df_copy.columns:
            data_target = df_copy[[self.target_feature]].values
        else:
            data_target = None
        features_scaled = self.feature_scaler.transform(data_features)
        # On crée les séquences avec la même longueur que lors de l'entraînement
        X, _ = self.create_sequences(features_scaled, features_scaled[:, 0])
        X = X.reshape(X.shape[0], X.shape[1], len(self.features))
        predictions = self.model.predict(X, verbose=0)
        predictions_inversed = self.target_scaler.inverse_transform(predictions)
        predictions_inversed += self.bias
        return predictions_inversed

    def auto_retrain(self, df, retrain_interval_hours=6):
        logger.info("Démarrage du réentraînement automatique du modèle...")
        self.train(df)
        logger.info("Réentraînement terminé.")

if __name__ == "__main__":
    # Exemple d'utilisation : entraînement initial à partir de "historique.csv"
    df = pd.read_csv("historique.csv")
    processor = DataProcessor()
    # Traitement des données brutes via la méthode process_data (sans changer l'ordre des colonnes)
    df = processor.process_data(df)
    model = ComplexPredictiveModel()
    model.train(df)

    # Réentraînement automatique toutes les 6 heures
    retrain_interval_seconds = 6 * 3600
    while True:
        print("Attente avant le prochain réentraînement...")
        time.sleep(retrain_interval_seconds)
        new_data = pd.read_csv("historique.csv")
        new_data = processor.process_data(new_data)
        print("Début du réentraînement automatique...")
        model.auto_retrain(new_data)
        print("Réentraînement terminé.")
