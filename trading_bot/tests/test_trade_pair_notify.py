import sys
import os
import unittest
from unittest.mock import patch

# Ajouter le chemin du projet au sys.path pour éviter les erreurs d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_bot.src.notifications.notify import send_trade_pair_notification

class TestTradePairNotify(unittest.TestCase):
    @patch('trading_bot.src.notifications.notify.send_trade_pair_notification', autospec=True)
    def test_send_trade_pair_notification(self, mock_notify):
        # Forcer le mock à intercepter l'appel et retourner un message spécifique
        mock_notify.return_value = "Notification envoyée avec succès"

        # Appel de la fonction (c'est le mock qui doit être utilisé)
        result = mock_notify(32000, 33000, 5, 5)

        # Vérifier que la fonction a bien été appelée avec les bons arguments
        mock_notify.assert_called_once_with(32000, 33000, 5, 5)

        # Vérifier que le retour est bien celui attendu
        self.assertEqual(result, "Notification envoyée avec succès", "La fonction doit retourner un message valide.")

if __name__ == '__main__':
    unittest.main()
