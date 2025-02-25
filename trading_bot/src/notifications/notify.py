import subprocess
import os

def send_trade_pair_notification(buy_price, sell_price, buy_fee, sell_fee, debug=True):
    """
    Envoie une notification Windows via notifu64.exe.

    :param buy_price: Prix d'achat
    :param sell_price: Prix de vente
    :param buy_fee: Frais d'achat
    :param sell_fee: Frais de vente
    :param debug: Active le mode debug pour afficher les logs
    :return: True si la notification est envoyée, False sinon
    """
    try:
        profit = (sell_price - buy_price) - (buy_fee + sell_fee)
        message = f"Achat: {buy_price}, Vente: {sell_price}, Profit: {profit} USD"

        if debug:
            print(f"[DEBUG] Notification message: {message}")

        # Définition du chemin vers notifu64.exe (doit se trouver dans le même répertoire que ce fichier)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        notifu_path = os.path.join(script_dir, "notifu64.exe")

        # Vérification si notifu64.exe est présent
        if not os.path.exists(notifu_path):
            raise FileNotFoundError(f"notifu64.exe non trouvé dans {notifu_path}")

        # Envoi de la notification via notifu64.exe
        subprocess.run([
            notifu_path,
            "/m", message,
            "/p", "Trade Bot",
            "/d", "5",
            "/i", "info"
        ], check=True)

        return True

    except Exception as e:
        print(f"Erreur lors de l'envoi de la notification: {e}")
        return False

if __name__ == "__main__":
    # Bloc de test : modification des valeurs si besoin pour vérifier le bon fonctionnement
    result = send_trade_pair_notification(32000, 33000, 5, 5, debug=True)
    print("Notification envoyée:", result)
