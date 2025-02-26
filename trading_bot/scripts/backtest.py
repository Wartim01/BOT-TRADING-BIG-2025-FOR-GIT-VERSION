import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model

# Ajouter le chemin du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.collector import DataCollector
from src.data.processor import DataProcessor
from src.strategies.technical import TechnicalStrategy
from src.risk.management import RiskManager
from src.utils.config_features import LSTM_FEATURES

# --------------------------- CONFIGURATION DU LOGGING ---------------------------
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(logs_dir, 'backtest.log'), encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# --------------------------- CHARGEMENT DE LA CONFIG ---------------------------
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info("Configuration loaded")
    return config

# --------------------------- CHARGEMENT DES HYPERPARAMÈTRES OPTIMAUX (PRODUCTION) ---------------------------
def load_best_hyperparameters(config):
    """
    Charge les hyperparamètres optimaux (production) depuis 'best_hyperparameters.pkl'
    et met à jour la configuration, y compris les paramètres de trading et de gestion du risque.
    """
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))
    best_hyper_path = os.path.join(model_dir, "best_hyperparameters.pkl")
    
    if os.path.exists(best_hyper_path):
        try:
            best_params = joblib.load(best_hyper_path)
            logger.info(f"Hyperparamètres optimaux chargés : {best_params}")
            # Mise à jour des paramètres de trading avec les valeurs optimisées
            config["signal_weights"]["lstm"] = best_params.get("lstm_weight", config["signal_weights"].get("lstm", 0.5))
            config["trading"]["buy_threshold"] = best_params.get("buy_threshold", config["trading"].get("buy_threshold", 0.05))
            config["trading"]["sell_threshold"] = best_params.get("sell_threshold", config["trading"].get("sell_threshold", -0.05))
            # Mise à jour des paramètres de gestion du risque
            if "risk_management" not in config["trading"]:
                config["trading"]["risk_management"] = {}
            config["trading"]["risk_management"]["max_risk_per_trade"] = best_params.get("max_risk_per_trade", config["trading"]["risk_management"].get("max_risk_per_trade", 0.05))
            config["trading"]["risk_management"]["stop_loss"] = best_params.get("stop_loss", config["trading"]["risk_management"].get("stop_loss", 0.02))
        except Exception as e:
            logger.error(f"Erreur lors du chargement des hyperparamètres optimaux : {e}")
    else:
        logger.info("Aucun fichier d'hyperparamètres optimaux trouvé. Utilisation des valeurs par défaut.")
    return config

# --------------------------- CALCUL DU SCORE COMPOSITE AVEC LSTM ---------------------------
def compute_composite_score_with_lstm(strategy, df_slice, weights):
    # Calcul du RSI
    rsi_series = strategy.compute_rsi(df_slice)
    if rsi_series.empty:
        raise ValueError("RSI calculé vide.")
    rsi_value = rsi_series.iloc[-1]
    rsi_signal = 1 if rsi_value < 30 else -1 if rsi_value > 70 else 0

    # Calcul du MACD
    macd_series, signal_series = strategy.compute_macd(df_slice)
    macd_signal = 0
    if len(macd_series) > 1:
        if macd_series.iloc[-2] < signal_series.iloc[-2] and macd_series.iloc[-1] > signal_series.iloc[-1]:
            macd_signal = 1
        elif macd_series.iloc[-2] > signal_series.iloc[-2] and macd_series.iloc[-1] < signal_series.iloc[-1]:
            macd_signal = -1

    # Bollinger Bands
    upper_band_series, lower_band_series = strategy.compute_bollinger_bands(df_slice)
    last_close = df_slice["close"].iloc[-1]
    if last_close <= lower_band_series.iloc[-1]:
        boll_signal = 1
    elif last_close >= upper_band_series.iloc[-1]:
        boll_signal = -1
    else:
        boll_signal = 0

    # Ichimoku
    conversion_line, base_line_series, leading_span_a, leading_span_b = strategy.compute_ichimoku(df_slice)
    if base_line_series.empty:
        raise ValueError("Ichimoku non calculable (base_line_series vide).")
    base_line = base_line_series.iloc[-1]
    ichimoku_signal = 1 if last_close > base_line else -1 if last_close < base_line else 0

    # ADX
    adx_series = strategy.compute_adx(df_slice)
    adx_value = adx_series.iloc[-1]
    adx_signal = 1 if adx_value > 25 else 0

    # Volume
    current_vol = df_slice["volume"].iloc[-1]
    if len(df_slice) >= 20:
        avg_vol = df_slice['volume'].rolling(window=20).mean().iloc[-1]
    else:
        avg_vol = current_vol
    if current_vol > avg_vol * 1.5:
        vol_signal = 1
    elif current_vol < avg_vol * 0.75:
        vol_signal = -1
    else:
        vol_signal = 0

    # Moyennes mobiles
    short_ma = df_slice['close'].rolling(window=50).mean().iloc[-1]
    if len(df_slice) >= 200:
        long_ma = df_slice['close'].rolling(window=200).mean().iloc[-1]
    else:
        long_ma = None
    ma_signal = 0 if long_ma is None else (1 if short_ma > long_ma else -1)

    tech_score = (
        rsi_signal * weights.get('rsi', 0.20) +
        macd_signal * weights.get('macd', 0.20) +
        boll_signal * weights.get('bollinger', 0.15) +
        ichimoku_signal * weights.get('ichimoku', 0.15) +
        adx_signal * weights.get('adx', 0.10) +
        vol_signal * weights.get('volume', 0.10) +
        ma_signal * weights.get('ma_crossover', 0.10)
    )

    # Calcul du signal LSTM
    predicted_price = strategy.predict_lstm(df_slice, sequence_length=60)
    if predicted_price is None:
        raise ValueError("Données insuffisantes pour la prédiction LSTM.")

    lstm_ratio = (predicted_price - last_close) / max(last_close, 1e-6)
    lstm_ratio = max(-1, min(1, lstm_ratio))
    lstm_weight = weights.get('lstm', 0.5)

    final_score = tech_score + lstm_weight * lstm_ratio
    return final_score

# --------------------------- FONCTION D'EXTRAPOLATION ---------------------------
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

# --------------------------- BACKTEST PRINCIPAL AVEC GESTION DES RISQUES ---------------------------
def run_backtest(config):
    mode = config["mode"]
    api_key = config["binance"][mode]["API_KEY"]
    api_secret = config["binance"][mode]["API_SECRET"]

    data_collector = DataCollector(api_key, api_secret, testnet=(mode=="testnet"))
    processor = DataProcessor()
    strategy = TechnicalStrategy()
    
    # Chargement des paramètres de trading et gestion des risques depuis la config
    initial_capital = config["trading"].get("initial_capital", 200)
    # Utilisation du paramètre optimisé pour le risque par trade (ex. 5% à 10% du capital)
    max_risk = config["trading"]["risk_management"].get("max_risk_per_trade", 0.05)
    stop_loss_pct = config["trading"]["risk_management"].get("stop_loss", 0.02)
    trading_fee = config["trading"].get("trading_fee", 0.001)
    
    # Création du RiskManager avec le risque maximal par trade
    risk_manager = RiskManager(account_balance=initial_capital, risk_percentage=max_risk)
    
    symbol = config.get("backtest", {}).get("symbol", "BTCUSDT")
    interval = config.get("backtest", {}).get("interval", "1m")
    start_str = config.get("backtest", {}).get("start_str", "7 days ago UTC")
    end_str = config.get("backtest", {}).get("end_str", None)
    
    buy_threshold = config["trading"].get("buy_threshold", 0.05)
    sell_threshold = config["trading"].get("sell_threshold", -0.05)
    signal_weights = config.get("signal_weights", {})
    
    logger.info(f"Récupération de l'historique {symbol} depuis '{start_str}' jusqu'à '{end_str}'...")
    raw_data = asyncio.run(data_collector.get_historical_data(symbol, interval, start_str, end_str))
    if raw_data is None:
        logger.error("Impossible de récupérer les données historiques.")
        return
    
    df = processor.process_data(raw_data)
    df.dropna(inplace=True)
    
    logger.info("Calcul des indicateurs techniques pour le LSTM...")
    df["rsi"] = strategy.compute_rsi(df)
    df["macd"], _ = strategy.compute_macd(df)
    df["bollinger_hband"], df["bollinger_lband"] = strategy.compute_bollinger_bands(df)
    df["adx"] = strategy.compute_adx(df)
    
    if len(df) < 200:
        logger.error("Historique insuffisant pour le backtest.")
        return
    
    position = 0.0
    entry_price = None
    trades = []
    capital = initial_capital
    max_capital = capital
    SEQ_LEN = 60  # Chaque séquence doit contenir au moins 60 lignes
    
    for i in range(200, len(df)):
        df_slice = df.iloc[i-SEQ_LEN:i]
        if len(df_slice) < SEQ_LEN:
            logger.warning(f"À l'index {i}, séquence incomplète. Extrapolation appliquée.")
            df_slice = extrapolate_dataframe(df_slice, SEQ_LEN, window=2)
    
        last_close = df["close"].iloc[i]
        
        try:
            final_score = compute_composite_score_with_lstm(strategy, df_slice, signal_weights)
        except ValueError as ve:
            logger.warning(f"Ignoré à l'index {i} : {ve}")
            continue
        
        # Si aucune position n'est ouverte, tenter d'ouvrir un trade
        if position == 0:
            if final_score >= buy_threshold:
                stop_loss_price = last_close * (1 - stop_loss_pct)
                position_size = risk_manager.calculate_position_size(last_close, stop_loss_price)
                if position_size > 0:
                    position = position_size
                    entry_price = last_close
                    logger.info(f"BUY @ {entry_price:.2f} | size={position_size:.6f} | stop_loss={stop_loss_price:.2f}")
            elif final_score <= sell_threshold:
                stop_loss_price = last_close * (1 + stop_loss_pct)
                position_size = risk_manager.calculate_position_size(last_close, stop_loss_price)
                if position_size > 0:
                    position = -position_size
                    entry_price = last_close
                    logger.info(f"SHORT @ {entry_price:.2f} | size={position_size:.6f} | stop_loss={stop_loss_price:.2f}")
        else:
            # Fermeture des positions
            if position > 0 and final_score <= sell_threshold:
                exit_price = last_close
                fee_buy = entry_price * abs(position) * trading_fee
                fee_sell = exit_price * abs(position) * trading_fee
                pnl = (exit_price - entry_price) * abs(position) - (fee_buy + fee_sell)
                capital += pnl
                trades.append({
                    "type": "BUY->SELL",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position,
                    "pnl": pnl
                })
                logger.info(f"SELL to close @ {exit_price:.2f} | size={position:.6f} | PnL={pnl:.2f} | capital={capital:.2f}")
                position = 0
                entry_price = None
            elif position < 0 and final_score >= buy_threshold:
                exit_price = last_close
                fee_buy = exit_price * abs(position) * trading_fee
                fee_sell = entry_price * abs(position) * trading_fee
                pnl = (entry_price - exit_price) * abs(position) - (fee_buy + fee_sell)
                capital += pnl
                trades.append({
                    "type": "SHORT->BUY",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position,
                    "pnl": pnl
                })
                logger.info(f"BUY to close short @ {exit_price:.2f} | size={position:.6f} | PnL={pnl:.2f} | capital={capital:.2f}")
                position = 0
                entry_price = None

        if capital > max_capital:
            max_capital = capital

    final_balance = capital
    gain = final_balance - initial_capital
    max_drawdown = ((max_capital - final_balance) / max_capital) * 100 if max_capital > 0 else 0.0
    winning_trades = [t for t in trades if t["pnl"] > 0]
    total_trades = len(trades)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0
    
    logger.info("======= Résultats du Backtest =======")
    logger.info(f"Nombre total de trades : {total_trades}")
    logger.info(f"Gains/Pertes (PNL) : {gain:.2f} (capital final : {final_balance:.2f})")
    logger.info(f"Taux de réussite : {win_rate:.2f}%")
    logger.info(f"Max drawdown : {max_drawdown:.2f}%")
    
    trades_df = pd.DataFrame(trades)
    results = {
        "total_trades": total_trades,
        "final_balance": final_balance,
        "pnl": gain,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trades_detail": trades_df
    }
    
    return results

# --------------------------- FONCTION MAIN ---------------------------
def main():
    try:
        config = load_config()
        # Charger les hyperparamètres optimaux de production depuis best_hyperparameters.pkl
        config = load_best_hyperparameters(config)
        results = run_backtest(config)
        if results:
            logger.info("Backtest terminé. Voici un résumé des résultats :")
            logger.info(results)
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du backtest : {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Backtest interrupted by the user. Exiting gracefully.")
