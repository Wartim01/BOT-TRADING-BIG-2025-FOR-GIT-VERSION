import pandas as pd
import requests
import time
import logging
import os
import sys

# Ajouter le dossier racine du projet dans le path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configuration de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# 📌 Configuration
# ----------------------------
SYMBOL = "BTCUSDT"    # Crypto à récupérer
INTERVAL = "1m"       # Intervalle de temps (1 minute)
LIMIT = 1000          # Nombre maximum de bougies par requête

# Pour récupérer les 120 dernières minutes (120 bougies)
TARGET_CANDLES = 120
logger.info(f"🎯 Objectif : récupérer {TARGET_CANDLES} bougies (15 minutes de données).")
# Rappel : Ce script devra être exécuté toutes les 20 minutes pour capter les données récentes.

def get_binance_data(symbol, interval, limit=1000, target_candles=TARGET_CANDLES):
    all_data = []
    end_time = int(time.time() * 1000)  # Timestamp actuel en ms
    session = requests.Session()

    while len(all_data) < target_candles:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 429:
                logger.warning("Taux de requêtes dépassé (429). Attente de 60 secondes...")
                time.sleep(60)
                continue

            response.raise_for_status()
            data = response.json()
            if not data or len(data) == 0:
                logger.error("🚨 Aucune donnée reçue, arrêt de la récupération.")
                break

            all_data.extend(data)
            # Mise à jour de end_time pour récupérer les données plus anciennes
            end_time = int(data[0][0]) - 1
            logger.info(f"✅ {len(all_data)} / {target_candles} bougies récupérées...")
            time.sleep(1)  # Pause pour éviter le ban API

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de la récupération des données : {e}")
            time.sleep(5)
            continue

    if not all_data:
        logger.error("Aucune donnée collectée.")
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
    logger.info("Démarrage de la récupération des données des 15 dernières minutes...")
    df = get_binance_data(SYMBOL, INTERVAL, LIMIT, TARGET_CANDLES)
    if not df.empty:
        # Limit to the 120 most recent candles pour permettre le calcul complet d'indicateurs (ex. ADX)
        df = df.tail(TARGET_CANDLES)
        output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/data/historique.csv"))
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Données enregistrées sous '{output_file}' avec {len(df)} lignes.")
    else:
        logger.error("❌ Échec de la récupération des données, fichier non enregistré.")

if __name__ == "__main__":
    main()
