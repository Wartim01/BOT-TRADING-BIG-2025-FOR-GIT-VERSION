import logging
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RiskManager:
    def __init__(self, account_balance=100, risk_percentage=0.01):
        """
        Initialise le gestionnaire de risque avec un capital de départ (en USDT) et
        le pourcentage du capital à risquer par trade (par défaut 1%).
        """
        self.account_balance = account_balance
        self.risk_percentage = risk_percentage
        self.trade_log = []

    def compute_atr(self, df, window=14):
        """
        Calcule l'Average True Range (ATR) pour mesurer la volatilité.
        :param df: DataFrame contenant les colonnes 'high', 'low', 'close'.
        :param window: Période pour le calcul de l'ATR.
        :return: Valeur de l'ATR.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low,
                        abs(high - close.shift(1)),
                        abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=window).mean().iloc[-1]
        return atr

    def calculate_position_size(self, entry_price, stop_loss_price, atr=None):
        """
        Calcule la taille de la position (en unités) en fonction du risque à supporter.
        Le risque par trade est défini comme un pourcentage du capital total.
        Si l'ATR est fourni et que la distance de stop loss est inférieure à l'ATR,
        cette dernière est utilisée pour ajuster le calcul.
        """
        risk_amount = self.account_balance * self.risk_percentage
        stop_loss_distance = abs(entry_price - stop_loss_price)
        if atr and stop_loss_distance < atr:
            logger.info(f"Utilisation de l'ATR pour ajuster la distance du stop loss: ATR={atr}, Distance initiale={stop_loss_distance}")
            stop_loss_distance = atr
        if stop_loss_distance == 0:
            logger.error("La distance du stop loss est zéro")
            return 0
        position_size = risk_amount / stop_loss_distance
        logger.info(f"Taille de position calculée : {position_size:.6f} unités (Risque = {risk_amount}, Distance = {stop_loss_distance})")
        return position_size

    def set_stop_loss(self, entry_price, risk_distance=None, atr=None):
        """
        Détermine le prix de stop loss.
        Si risk_distance n'est pas fourni, il est fixé à 1% du prix d'entrée ou à la valeur ATR si disponible.
        """
        if risk_distance is None:
            risk_distance = atr if atr else entry_price * self.risk_percentage
        stop_loss = entry_price - risk_distance
        logger.info(f"Stop loss fixé à : {stop_loss:.2f} (Distance = {risk_distance:.2f})")
        return stop_loss

    def trailing_stop(self, current_price, trailing_distance):
        """
        Ajuste le stop loss en mode trailing.
        """
        stop_loss = current_price - trailing_distance
        logger.info(f"Trailing stop ajusté à : {stop_loss:.2f} (Distance = {trailing_distance:.2f})")
        return stop_loss

    def record_trade(self, trade):
        """
        Enregistre le trade dans le log et met à jour le capital en fonction du profit réalisé.
        """
        self.trade_log.append(trade)
        profit = trade.get('profit', 0)
        self.update_balance(profit)
        logger.info(f"Trade enregistré : {trade}")
        logger.info(f"Nouveau solde : {self.account_balance:.2f} USDT")

    def update_balance(self, profit):
        """
        Met à jour le capital en ajoutant le profit (ou en retranchant la perte).
        """
        self.account_balance += profit

    def generate_report(self):
        """
        Génère un rapport synthétique des trades.
        """
        total_trades = len(self.trade_log)
        winning_trades = sum(1 for trade in self.trade_log if trade['profit'] > 0)
        losing_trades = total_trades - winning_trades
        total_profit = sum(trade['profit'] for trade in self.trade_log)
        
        balance_history = [self.account_balance]
        running_balance = self.account_balance
        for trade in self.trade_log:
            running_balance += trade.get('profit', 0)
            balance_history.append(running_balance)
        peak = max(balance_history)
        drawdown = ((peak - min(balance_history)) / peak) * 100 if peak > 0 else 0
        
        report = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit': total_profit,
            'final_balance': self.account_balance,
            'max_drawdown': drawdown
        }
        logger.info(f"Rapport généré : {report}")
        return report

    def adjust_risk_percentage(self, recent_performance, current_volatility=None, signal_strength=None, min_risk_amount=5, max_risk_amount=20):
        """
        Ajuste dynamiquement le pourcentage de risque par trade en fonction des performances récentes,
        de la volatilité du marché et de la force du signal technique, tout en garantissant qu'au moins
        min_risk_amount USDT sont risqués et que le risque ne dépasse pas max_risk_amount USDT.
        
        :param recent_performance: Dictionnaire contenant par exemple 'total_profit'.
        :param current_volatility: Valeur de volatilité actuelle (ex: ATR relatif, par exemple 0.05 pour 5%).
        :param signal_strength: Force du signal technique (valeur entre 0 et 1).
        :param min_risk_amount: Montant minimum en USDT à risquer par trade (par défaut 5 USDT).
        :param max_risk_amount: Montant maximum en USDT à risquer par trade (par défaut 20 USDT).
        :return: Nouveau risque en pourcentage.
        """
        base_risk = self.risk_percentage
        if recent_performance.get('total_profit', 0) < 0:
            base_risk *= 0.9
            logger.info(f"Performance négative: réduction du risque à {base_risk*100:.2f}%")
        else:
            base_risk *= 1.05
            logger.info(f"Performance positive: augmentation du risque à {base_risk*100:.2f}%")
        
        if current_volatility is not None:
            volatility_threshold_high = 0.05  # 5% ATR relatif
            volatility_threshold_low = 0.02   # 2% ATR relatif
            if current_volatility > volatility_threshold_high:
                volatility_factor = 0.8
                logger.info(f"Volatilité élevée ({current_volatility:.2f}): facteur de risque {volatility_factor}")
            elif current_volatility < volatility_threshold_low:
                volatility_factor = 1.2
                logger.info(f"Volatilité faible ({current_volatility:.2f}): facteur de risque {volatility_factor}")
            else:
                volatility_factor = 1.0
            base_risk *= volatility_factor

        if signal_strength is not None:
            if signal_strength > 0.8:
                signal_factor = 1.1
                logger.info(f"Signal fort ({signal_strength:.2f}): facteur de risque {signal_factor}")
            elif signal_strength < 0.5:
                signal_factor = 0.9
                logger.info(f"Signal faible ({signal_strength:.2f}): facteur de risque {signal_factor}")
            else:
                signal_factor = 1.0
            base_risk *= signal_factor

        # Calcul du montant risqué basé sur le risque ajusté
        computed_risk_amount = self.account_balance * base_risk
        if computed_risk_amount < min_risk_amount:
            adjusted_risk = min_risk_amount / self.account_balance
            logger.info(f"Montant risqué {computed_risk_amount:.2f} USDT inférieur au minimum ({min_risk_amount} USDT). Ajustement du risque à {adjusted_risk*100:.2f}%.")
            base_risk = adjusted_risk
        elif computed_risk_amount > max_risk_amount:
            adjusted_risk = max_risk_amount / self.account_balance
            logger.info(f"Montant risqué {computed_risk_amount:.2f} USDT supérieur au maximum ({max_risk_amount} USDT). Ajustement du risque à {adjusted_risk*100:.2f}%.")
            base_risk = adjusted_risk

        # Limiter le risque entre min_risk_amount/account_balance et max_risk_amount/account_balance
        new_risk = max(min_risk_amount / self.account_balance, min(base_risk, max_risk_amount / self.account_balance))
        self.risk_percentage = new_risk
        logger.info(f"Risque ajusté final: {self.risk_percentage*100:.2f}% (soit {self.account_balance * self.risk_percentage:.2f} USDT risqués)")
        return self.risk_percentage
