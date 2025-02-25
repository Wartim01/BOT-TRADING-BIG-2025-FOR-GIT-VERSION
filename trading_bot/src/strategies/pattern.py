import pandas as pd
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PatternStrategy:
    def detect_doji(self, df, threshold=0.1):
        start_time = time.time()
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['doji'] = (df['body'] / df['range']) < threshold
        result = df[df['doji']]
        duration = time.time() - start_time
        logger.info(f"detect_doji executed in {duration:.4f} seconds")
        return result

    def detect_hammer(self, df, factor=2):
        start_time = time.time()
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['hammer'] = (df['lower_shadow'] > factor * df['body']) & (df['upper_shadow'] < df['body'])
        result = df[df['hammer']]
        duration = time.time() - start_time
        logger.info(f"detect_hammer executed in {duration:.4f} seconds")
        return result
