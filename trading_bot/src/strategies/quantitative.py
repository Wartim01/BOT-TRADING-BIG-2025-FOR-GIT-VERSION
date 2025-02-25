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
        data_features = df[self.features].values
        data_target = df[[self.target_feature]].values
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
        data_features = df[self.features].values
        data_target = df[[self.target_feature]].values
        features_scaled = self.feature_scaler.transform(data_features)
        target_scaled = self.target_scaler.transform(data_target)
        X, _ = self.create_sequences(features_scaled, target_scaled)
        X = X.reshape(X.shape[0], X.shape[1], len(self.features))
        predictions = self.model.predict(X, verbose=0)
        predictions_inversed = self.target_scaler.inverse_transform(predictions)
        predictions_inversed += self.bias
        return predictions_inversed
