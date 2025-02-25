# src/utils/config.py
import os
import json

def load_config():
    # Chemin du fichier de configuration (à adapter selon l'arborescence)
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remplacement des clés sensibles par celles définies dans les variables d'environnement.
    # Si la variable d'environnement n'est pas définie, la valeur par défaut du fichier sera utilisée.
    config['binance']['production']['API_KEY'] = os.environ.get("BINANCE_PROD_API_KEY", config['binance']['production']['API_KEY'])
    config['binance']['production']['API_SECRET'] = os.environ.get("BINANCE_PROD_API_SECRET", config['binance']['production']['API_SECRET'])
    config['binance']['testnet']['API_KEY'] = os.environ.get("BINANCE_TEST_API_KEY", config['binance']['testnet']['API_KEY'])
    config['binance']['testnet']['API_SECRET'] = os.environ.get("BINANCE_TEST_API_SECRET", config['binance']['testnet']['API_SECRET'])
    
    return config
