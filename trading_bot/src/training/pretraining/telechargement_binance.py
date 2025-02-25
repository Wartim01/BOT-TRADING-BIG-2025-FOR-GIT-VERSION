import pandas as pd
import requests
import time
import logging
import os

# Configuration de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# 📌 Configuration
# ----------------------------
SYMBOL = "BTCUSDT"    # Crypto à récupérer
INTERVAL = "1m"       # Intervalle de temps (ex: "1m", "5m", "1h", "1d")
LIMIT = 1000          # Nombre de bougies max par requête
HISTORICAL_DAYS = 30  # Nombre de jours d’historique à récupérer

# Calcul du nombre de bougies cibles en fonction de l'intervalle
try:
    interval_minutes = int(INTERVAL[:-1])
except ValueError:
    logging.error("Intervalle incorrect. Format attendu par exemple '1m'.")
    raise

TARGET_CANDLES = (HISTORICAL_DAYS * 24 * 60) // interval_minutes
logging.info(f"🎯 Objectif : récupérer {TARGET_CANDLES} bougies.")

def get_binance_data(symbol, interval, limit=1000, target_candles=TARGET_CANDLES):
    all_data = []
    end_time = int(time.time() * 1000)  # Timestamp actuel en ms
    session = requests.Session()

    while len(all_data) < target_candles:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 429:
                logging.warning("Taux de requêtes dépassé (429). Attente de 60 secondes...")
                time.sleep(60)
                continue
            response.raise_for_status()
            data = response.json()
            if not data or len(data) == 0:
                logging.error("🚨 Aucune donnée reçue, arrêt de la récupération.")
                break
            all_data.extend(data)
            # Mise à jour de end_time pour éviter les doublons
            end_time = int(data[0][0]) - 1
            logging.info(f"✅ {len(all_data)} / {target_candles} bougies récupérées...")
            time.sleep(1)  # Pause pour éviter le ban API
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Erreur lors de la récupération des données : {e}")
            time.sleep(5)
            continue

    if not all_data:
        logging.error("Aucune donnée collectée.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    # Conversion du timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Sélection des colonnes utiles
    df = df[["timestamp", "close", "high", "low", "volume"]]
    # Conversion des valeurs en float
    for col in ["close", "high", "low", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values(by="timestamp").reset_index(drop=True)

def main():
    df = get_binance_data(SYMBOL, INTERVAL, LIMIT, TARGET_CANDLES)
    if not df.empty:
        df = df.tail(TARGET_CANDLES)
        # Update the output directory to avoid duplicate "src" in the path
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "historique_pre-entrainement.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"✅ Données enregistrées sous '{output_file}' avec {len(df)} lignes.")
    else:
        logging.error("❌ Échec de la récupération des données, fichier non enregistré.")

if __name__ == "__main__":
    main()
