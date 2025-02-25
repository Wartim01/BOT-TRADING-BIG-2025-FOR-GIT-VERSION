from binance.client import Client
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class OrderManager:
    def __init__(self, api_key, api_secret, testnet=False):
        """
        Initialise la connexion à Binance.
        """
        try:
            self.client = Client(api_key, api_secret, testnet=testnet)
            logger.info("Connexion à Binance établie avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à Binance: {str(e)}")
            raise

    def place_order(self, symbol, side, order_type, quantity, price=None, stop_loss=0.02, take_profit=0.05):
        """
        Place un ordre avec gestion de stop-loss et take-profit.

        :param symbol: Symbole de l'actif (ex. 'BTCUSDT').
        :param side: 'BUY' ou 'SELL'.
        :param order_type: Type d'ordre (ex. 'MARKET', 'LIMIT', etc.).
        :param quantity: Quantité à trader.
        :param price: Prix d'exécution (obligatoire pour un ordre LIMIT).
        :param stop_loss: Pourcentage de stop loss.
        :param take_profit: Pourcentage de take profit.
        :return: Réponse de l'API ou None en cas d'erreur.
        """
        if order_type.upper() == 'LIMIT' and price is None:
            logger.error("Le prix doit être spécifié pour un ordre LIMIT.")
            return None

        try:
            side = side.upper()
            order_type = order_type.upper()
            
            # Calcul des niveaux de stop loss et take profit si le prix est fourni
            if price is not None:
                if side == 'BUY':
                    sl_price = price * (1 - stop_loss)
                    tp_price = price * (1 + take_profit)
                elif side == 'SELL':
                    sl_price = price * (1 + stop_loss)
                    tp_price = price * (1 - take_profit)
                else:
                    logger.error(f"Côté d'ordre non valide: {side}")
                    return None
            else:
                sl_price, tp_price = None, None

            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity,
            }
            if price is not None:
                order_params['price'] = price

            if order_type in ['STOP_LOSS_LIMIT', 'STOP_LOSS', 'TAKE_PROFIT_LIMIT', 'TAKE_PROFIT']:
                order_params['stopPrice'] = sl_price
                order_params['stopLimitPrice'] = tp_price

            logger.info(f"Passage de l'ordre: {order_params}")
            order_response = self.client.create_order(**order_params)

            if 'orderId' in order_response:
                logger.info(f"Ordre passé avec succès : Order ID {order_response['orderId']}")
                return order_response
            else:
                logger.error(f"Réponse inattendue de l'API: {order_response}")
                return None

        except Exception as e:
            logger.error(f"Erreur lors du passage de l'ordre: {str(e)}")
            return None

    def monitor_order(self, symbol, order_id, check_interval=5, timeout=60):
        """
        Surveille l'état d'un ordre passé et renvoie son état final.
        
        :param symbol: Symbole de l'actif.
        :param order_id: ID de l'ordre à surveiller.
        :param check_interval: Intervalle (en secondes) entre chaque vérification.
        :param timeout: Durée maximale (en secondes) d'attente.
        :return: L'état final de l'ordre ou None en cas de timeout.
        """
        start_time = time.time()
        while True:
            try:
                order_status = self.client.get_order(symbol=symbol, orderId=order_id)
                status = order_status.get("status", "")
                logger.info(f"État de l'ordre {order_id}: {status}")
                if status in ["FILLED", "CANCELED", "EXPIRED"]:
                    return order_status
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de l'ordre {order_id}: {str(e)}")
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout atteint pour l'ordre {order_id}.")
                return None
            time.sleep(check_interval)
