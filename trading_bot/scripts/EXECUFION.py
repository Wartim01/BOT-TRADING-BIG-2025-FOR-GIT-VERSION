import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.src.data.collector import DataCollector
from trading_bot.src.data.processor import DataProcessor
from trading_bot.src.strategies.technical import TechnicalStrategy
from trading_bot.src.risk.management import RiskManager

# ====================== CONFIGURATION DU LOGGING ======================
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

# ====================== CHARGEMENT DE LA CONFIG ======================
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration introuvable: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info("Configuration loaded")
    return config

# ====================== CALCUL DU SCORE COMPOSITE ======================
def compute_composite_score_with_lstm(strategy, df_slice, weights):
    # Score technique
    SEQ_LEN = 60
    if len(df_slice) < SEQ_LEN:
        pad_rows = SEQ_LEN - len(df_slice)
        padding = df_slice.iloc[[-1]].copy().repeat(pad_rows)
        df_slice = pd.concat([df_slice, padding], ignore_index=True)
        logger.warning("Padding appliqué pour constituer une séquence complète.")

    rsi_value = strategy.compute_rsi(df_slice).iloc[-1]
    rsi_signal = 1 if rsi_value < 30 else -1 if rsi_value > 70 else 0
    
    macd_series, signal_series = strategy.compute_macd(df_slice)
    macd_signal = 0
    if len(macd_series) > 1:
        if macd_series.iloc[-2] < signal_series.iloc[-2] and macd_series.iloc[-1] > signal_series.iloc[-1]:
            macd_signal = 1
        elif macd_series.iloc[-2] > signal_series.iloc[-2] and macd_series.iloc[-1] < signal_series.iloc[-1]:
            macd_signal = -1

    upper_band_series, lower_band_series = strategy.compute_bollinger_bands(df_slice)
    upper_band = upper_band_series.iloc[-1]
    lower_band = lower_band_series.iloc[-1]
    last_close = df_slice["close"].iloc[-1]
    if last_close <= lower_band:
        boll_signal = 1
    elif last_close >= upper_band:
        boll_signal = -1
    else:
        boll_signal = 0

    # Unpacking de 4 valeurs pour Ichimoku
    conversion_line, base_line_series, leading_span_a, leading_span_b = strategy.compute_ichimoku(df_slice)
    base_line = base_line_series.iloc[-1]
    ichimoku_signal = 1 if last_close > base_line else -1 if last_close < base_line else 0

    adx_series = strategy.compute_adx(df_slice)
    adx_value = adx_series.iloc[-1]
    adx_signal = 1 if adx_value > 25 else 0

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

    # Signal LSTM
    predicted_price = strategy.predict_lstm(df_slice, sequence_length=SEQ_LEN)
    if predicted_price is None:
        raise ValueError("LSTM indisponible : Bot stoppé (pas de fallback).")

    lstm_ratio = (predicted_price - last_close) / max(last_close, 1e-6)
    lstm_ratio = max(-1, min(1, lstm_ratio))
    lstm_weight = weights.get('lstm', 0.5)

    final_score = tech_score + lstm_weight * lstm_ratio
    return final_score

# ====================== BACKTEST PRINCIPAL ======================
def run_backtest(config):
    mode = config["mode"]
    api_key = config["binance"][mode]["API_KEY"]
    api_secret = config["binance"][mode]["API_SECRET"]

    data_collector = DataCollector(api_key, api_secret, testnet=(mode=="testnet"))
    processor = DataProcessor()
    strategy = TechnicalStrategy()
    
    initial_capital = config["trading"].get("initial_capital", 200)
    risk_pct = config["trading"].get("risk_percentage", 0.01)
    risk_manager = RiskManager(account_balance=initial_capital, risk_percentage=risk_pct)
    
    # Paramètres du backtest
    symbol = config.get("backtest", {}).get("symbol", "BTCUSDT")
    interval = config.get("backtest", {}).get("interval", "1m")
    start_str = config.get("backtest", {}).get("start_str", "7 days ago UTC")
    end_str = config.get("backtest", {}).get("end_str", None)
    
    buy_threshold = config["trading"].get("buy_threshold", 0.5)
    sell_threshold = config["trading"].get("sell_threshold", -0.5)
    signal_weights = config.get("signal_weights", {})
    
    logger.info(f"Récupération de l'historique {symbol} depuis '{start_str}' jusqu'à '{end_str}'...")
    raw_data = asyncio.run(data_collector.get_historical_data(symbol, interval, start_str, end_str))
    if raw_data is None:
        logger.error("Impossible de récupérer les données historiques.")
        return
    
    df = processor.process_data(raw_data)
    df.dropna(inplace=True)
    
    # Ajout des indicateurs pour le LSTM
    logger.info("Calcul des indicateurs techniques pour le LSTM...")
    df["rsi"] = strategy.compute_rsi(df)
    df["macd"], _ = strategy.compute_macd(df)
    df["bollinger_hband"], df["bollinger_lband"] = strategy.compute_bollinger_bands(df)
    df["adx"] = strategy.compute_adx(df)
    # Autres indicateurs peuvent être ajoutés ici

    if len(df) < 200:
        logger.error("Historique insuffisant pour le backtest.")
        return
    
    position = 0.0
    entry_price = None
    trades = []
    capital = initial_capital
    max_capital = capital
    
    for i in range(200, len(df)):
        df_slice = df.iloc[i-60:i]
        last_close = df["close"].iloc[i]
        
        try:
            final_score = compute_composite_score_with_lstm(strategy, df_slice, signal_weights)
        except ValueError as ve:
            logger.warning(f"LSTM indisponible à l'index {i}: {ve}")
            continue
        
        # Simulation du trade
        if position == 0:
            if final_score >= buy_threshold:
                stop_loss_price = last_close * (1 - config["trading"]["risk_management"]["stop_loss"])
                position_size = risk_manager.calculate_position_size(last_close, stop_loss_price)
                if position_size > 0:
                    position = position_size
                    entry_price = last_close
                    logger.info(f"BUY @ {entry_price} | size={position_size}")
            elif final_score <= sell_threshold:
                stop_loss_price = last_close * (1 + config["trading"]["risk_management"]["stop_loss"])
                position_size = risk_manager.calculate_position_size(last_close, stop_loss_price)
                if position_size > 0:
                    position = -position_size
                    entry_price = last_close
                    logger.info(f"SHORT @ {entry_price} | size={position_size}")
        else:
            if position > 0 and final_score <= sell_threshold:
                exit_price = last_close
                pnl = (exit_price - entry_price) * abs(position)
                capital += pnl
                trades.append({
                    "type": "BUY->SELL",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position,
                    "pnl": pnl
                })
                logger.info(f"SELL to close @ {exit_price} | size={position} | PnL={pnl:.2f} | capital={capital:.2f}")
                position = 0
                entry_price = None
            elif position < 0 and final_score >= buy_threshold:
                exit_price = last_close
                pnl = (entry_price - exit_price) * abs(position)
                capital += pnl
                trades.append({
                    "type": "SHORT->BUY",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position,
                    "pnl": pnl
                })
                logger.info(f"BUY to close short @ {exit_price} | size={position} | PnL={pnl:.2f} | capital={capital:.2f}")
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

def main():
    config = load_config()
    results = run_backtest(config)
    if results:
        logger.info("Backtest terminé. Voici un résumé des résultats :")
        logger.info(results)

if __name__ == "__main__":
    main()
