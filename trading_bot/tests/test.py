import joblib
scaler_path = "C:\\Users\\timot\\OneDrive\\Bureau\\BOT TRADING BIG 2025\\trading_bot\\src\\models\\scaler.pkl"
scaler = joblib.load(scaler_path)
print("Scaler chargé :", scaler)
