import sys
import os
import unittest
import numpy as np
import pandas as pd
import logging

# Ajouter le chemin du projet au sys.path pour éviter les erreurs d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from trading_bot.src.strategies.quantitative import ComplexPredictiveModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TestOptimalPredictiveModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Génération d'un jeu de données synthétique réaliste avec 1000 lignes
        np.random.seed(42)
        rows = 1000
        dates = pd.date_range("2020-01-01", periods=rows, freq="H")
        data = {
            "open": np.random.uniform(9000, 10000, rows),
            "high": np.random.uniform(10000, 11000, rows),
            "low": np.random.uniform(8000, 9000, rows),
            "close": np.random.uniform(9000, 10000, rows),
            "volume": np.random.uniform(100, 1000, rows)
        }
        cls.df = pd.DataFrame(data, index=dates)
        # Instanciation du modèle avec des paramètres pour accélérer les tests
        cls.model = ComplexPredictiveModel(sequence_length=100, epochs=20, batch_size=16)
        cls.model.train(cls.df)
        cls.predictions = cls.model.predict(cls.df)

    def test_predictions_shape(self):
        expected_length = len(self.df) - self.model.sequence_length
        self.assertEqual(self.predictions.shape, (expected_length, 1),
                         "La forme des prédictions n'est pas conforme.")

    def test_bias_calibration(self):
        self.assertFalse(np.isnan(self.model.bias), "Le biais de calibration est NaN.")

    def test_prediction_values_positive(self):
        self.assertTrue(np.all(self.predictions > 0), "Certaines prédictions sont négatives ou nulles.")

    def test_error_metrics(self):
        actual = self.df["close"].values[self.model.sequence_length:]
        mae = np.mean(np.abs(actual - self.predictions.flatten()))
        rmse = np.sqrt(np.mean((actual - self.predictions.flatten())**2))
        logger.info(f"MAE: {mae}, RMSE: {rmse}")
        # Seuils définis pour le jeu de données synthétique
        self.assertTrue(mae < 500, "Le MAE est trop élevé pour un modèle optimal.")
        self.assertTrue(rmse < 600, "Le RMSE est trop élevé pour un modèle optimal.")

if __name__ == "__main__":
    unittest.main()
