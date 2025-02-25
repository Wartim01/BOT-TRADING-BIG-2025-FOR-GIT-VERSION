import logging
import pandas as pd
import numpy as np
import time

from trading_bot.src.data.collector import DataCollector
from trading_bot.src.data.processor import DataProcessor
from trading_bot.src.strategies.technical import TechnicalStrategy
from trading_bot.src.strategies.quantitative import QuantitativeStrategy
from trading_bot.src.strategies.pattern import PatternStrategy
from trading_bot.src.risk.management import RiskManager
from trading_bot.src.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TradeController:
    def __init__(self, config):
        """
        Initialise tous les modules nécessaires : collecte, traitement, stratégies, gestion du risque et exécution.
        """
        self.config = config
        self.symbol = "BTCUSDT"
        mode = config["mode"]
        self.collector = DataCollector(
            config["binance"][mode]["API_KEY"],
            config["binance"][mode]["API_SECRET"],
            testnet=(mode=="testnet")
        )
        self.processor = DataProcessor()
        self.tech_strategy = TechnicalStrategy()
        self.quant_strategy = QuantitativeStrategy()
        self.pattern_strategy = PatternStrategy()
        self.risk_manager = RiskManager(
            account_balance=config["trading"]["initial_capital"],
            risk_percentage=config["trading"].get("risk_management", {}).get("stop_loss", 0.02)
        )
        self.order_manager = OrderManager(
            config["binance"][mode]["API_KEY"],
            config["binance"][mode]["API_SECRET"],
            testnet=(mode=="testnet")
        )
        # Seuils de décision pour les signaux (exemples)
        self.buy_threshold = 0.5
        self.sell_threshold = -0.5

    def get_latest_data(self, interval='1m', lookback='90 minutes ago UTC'):
        """
        Récupère les données historiques récentes et les formate en DataFrame.
        """
        data = self.collector.get_historical_data(self.symbol, interval, lookback)
        if isinstance(data, list):
            columns = [
                'timestamp','open','high','low','close','volume',
                'close_time','quote_asset_volume','number_of_trades',
                'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ]
            df = pd.DataFrame(data, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            return df
        else:
            logger.error("Erreur lors de la récupération des données.")
            return None

    def generate_composite_signal(self, df):
        """
        Combine les signaux issus des différentes stratégies pour produire un signal composite.
        """
        # Assurer la cohérence des colonnes en minuscules
        df.columns = [col.lower() for col in df.columns]

        # Calcul des indicateurs techniques
        df['rsi'] = self.tech_strategy.compute_rsi(df)
        macd, signal = self.tech_strategy.compute_macd(df)
        df['macd'] = macd
        df['signal'] = signal
        upper_band, lower_band = self.tech_strategy.compute_bollinger_bands(df)
        df['bollinger_hband'] = upper_band
        df['bollinger_lband'] = lower_band

        # Signal technique basé sur RSI
        latest_rsi = df['rsi'].iloc[-1]
        rsi_signal = 1 if latest_rsi < 30 else (-1 if latest_rsi > 70 else 0)

        # Signal MACD : détection de croisement
        if len(df) >= 2:
            macd_prev, signal_prev = df['macd'].iloc[-2], df['signal'].iloc[-2]
            macd_curr, signal_curr = df['macd'].iloc[-1], df['signal'].iloc[-1]
            if macd_prev < signal_prev and macd_curr > signal_curr:
                macd_signal = 1
            elif macd_prev > signal_prev and macd_curr < signal_curr:
                macd_signal = -1
            else:
                macd_signal = 0
        else:
            macd_signal = 0

        # Signal pattern : détection de Doji
        doji_df = self.pattern_strategy.detect_doji(df)
        pattern_signal = 1 if not doji_df.empty else 0

        # Signal quantitatif : tendance sur 30 périodes
        trend_df = self.quant_strategy.compute_trend(df, window=30)
        trend = trend_df['trend'].iloc[-1] if 'trend' in trend_df.columns and not trend_df['trend'].isna().all() else 0
        quant_signal = 1 if trend > 0 else (-1 if trend < 0 else 0)

        # Pondération des signaux (exemple de pondérations)
        composite_signal = 0.4 * rsi_signal + 0.3 * macd_signal + 0.1 * pattern_signal + 0.2 * quant_signal
        logger.info(f"Composite signal: {composite_signal} (RSI: {rsi_signal}, MACD: {macd_signal}, Pattern: {pattern_signal}, Trend: {quant_signal})")
        return composite_signal

    def execute_trade(self, signal):
        """
        Exécute un ordre d'achat ou de vente en fonction du signal composite.
        """
        df_recent = self.get_latest_data(interval='1m', lookback='5 minutes ago UTC')
        if df_recent is None or df_recent.empty:
            logger.error("Pas de données récentes pour obtenir le prix actuel.")
            return

        current_price = df_recent['close'].iloc[-1]
        logger.info(f"Prix actuel: {current_price}")

        # Détermination de l'action selon le signal
        if signal >= self.buy_threshold:
            side = 'BUY'
        elif signal <= self.sell_threshold:
            side = 'SELL'
        else:
            logger.info("Signal neutre, aucune action effectuée.")
            return

        # Récupérer des données plus longues pour calculer l'ATR
        df_atr = self.get_latest_data(interval='1m', lookback='60 minutes ago UTC')
        if df_atr is None or df_atr.empty:
            logger.error("Pas de données pour le calcul de l'ATR.")
            return

        atr = self.risk_manager.compute_atr(df_atr, window=14)
        # Définir le stop loss avec l'ATR
        stop_loss = self.risk_manager.set_stop_loss(current_price, atr=atr)
        # Calculer la taille de position
        position_size = self.risk_manager.calculate_position_size(current_price, stop_loss, atr=atr)

        logger.info(f"Préparation de l'ordre {side} : taille = {position_size}, prix = {current_price}, stop_loss = {stop_loss}")
        # Passer l'ordre via OrderManager (ici ordre LIMIT pour l'exemple)
        order_response = self.order_manager.place_order(
            symbol=self.symbol,
            side=side,
            order_type="LIMIT",
            quantity=position_size,
            price=current_price
        )
        if order_response:
            trade = {
                'id': order_response.get('orderId', None),
                'timestamp': str(pd.Timestamp.now()),
                'asset': self.symbol,
                'type': side,
                'quantity': position_size,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': current_price * (1.05 if side == 'BUY' else 0.95),
                'profit': 0  # À mettre à jour lors de la clôture du trade
            }
            self.risk_manager.record_trade(trade)
        else:
            logger.error("Échec du placement de l'ordre.")

    def run(self):
        """
        Exécute le pipeline complet :
        - Récupération des données récentes.
        - Génération du signal composite.
        - Exécution de l'ordre si le signal est significatif.
        """
        df = self.get_latest_data()
        if df is None or df.empty:
            logger.error("Pas de données disponibles pour générer un signal.")
            return

        composite_signal = self.generate_composite_signal(df)
        self.execute_trade(composite_signal)

if __name__ == "__main__":
    # Exemple de configuration (à adapter selon votre environnement)
    config = {
        "binance": {
            "production": {
                "API_KEY": "prod_api_key_placeholder",
                "API_SECRET": "prod_api_secret_placeholder"
            },
            "testnet": {
                "API_KEY": "test_api_key_placeholder",
                "API_SECRET": "test_api_secret_placeholder"
            }
        },
        "mode": "testnet",
        "trading": {
            "initial_capital": 200,
            "trading_fee": 0.001,
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.05
            }
        }
    }
    controller = TradeController(config)
    # Exécution continue toutes les 60 secondes
    while True:
        controller.run()
        time.sleep(60)
