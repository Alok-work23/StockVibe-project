import joblib

model = joblib.load("../models/stock_predictor.pkl")

def predict_stock(features):
    prediction = model.predict([features])[0]

    return prediction