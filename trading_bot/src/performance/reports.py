import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PerformanceReports:
    def __init__(self, trades):
        self.trades = trades
    def generate_summary(self):
        logger.info("Generating performance summary")
        df = pd.DataFrame(self.trades)
        total_trades = len(df)
        total_profit = df['profit'].sum()
        average_profit = df['profit'].mean() if total_trades > 0 else 0
        summary = {
            "total_trades": total_trades,
            "total_profit": total_profit,
            "average_profit": average_profit
        }
        logger.info("Performance summary: %s", summary)
        return summary
