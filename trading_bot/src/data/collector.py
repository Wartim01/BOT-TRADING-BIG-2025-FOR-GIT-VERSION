import time
import logging
import asyncio
from binance.client import Client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DataCollector:
    def __init__(self, api_key, api_secret, testnet=False, max_retries=5, backoff_factor=2, cache_ttl=60):
        """
        Initialise le DataCollector.
        
        :param api_key: Clé API Binance.
        :param api_secret: Secret API Binance.
        :param testnet: Si True, se connecte au testnet.
        :param max_retries: Nombre maximal de tentatives en cas d'erreur.
        :param backoff_factor: Facteur de temps d'attente exponentiel entre chaque tentative.
        :param cache_ttl: Durée de vie du cache (en secondes) pour les appels répétitifs.
        """
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.cache = {}  # Structure de cache : {clé: (résultat, timestamp)}
        self.cache_ttl = cache_ttl

    def _cache_key(self, symbol, interval, start_str, end_str):
        """Génère une clé de cache unique pour une requête donnée."""
        return f"{symbol}_{interval}_{start_str}_{end_str}"

    async def get_historical_data(self, symbol, interval, start_str, end_str=None):
        """
        Récupère les données historiques de Binance de façon asynchrone.
        Utilise un cache pour limiter les appels répétitifs.

        :param symbol: Symbole de la crypto (ex. 'BTCUSDT').
        :param interval: Intervalle (ex. '1m').
        :param start_str: Date de début (ex. '90 minutes ago UTC').
        :param end_str: (Optionnel) Date de fin.
        :return: Liste des bougies ou None en cas d'erreur.
        """
        key = self._cache_key(symbol, interval, start_str, end_str)
        now = time.time()

        # Vérification du cache
        if key in self.cache:
            cached_result, timestamp = self.cache[key]
            if now - timestamp < self.cache_ttl:
                logger.info("Utilisation des données mises en cache.")
                return cached_result

        attempt = 0
        result = None
        while attempt < self.max_retries:
            try:
                # Exécution asynchrone de la requête via un thread séparé
                result = await asyncio.to_thread(
                    self.client.get_historical_klines, symbol, interval, start_str, end_str
                )
                # Stockage dans le cache
                self.cache[key] = (result, now)
                return result
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                await asyncio.sleep(self.backoff_factor ** attempt)
                attempt += 1

        logger.error("Max retries exceeded")
        return None
