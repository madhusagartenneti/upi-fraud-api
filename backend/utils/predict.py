import joblib
from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH,
    SCALER_PATH,
    DEVICE_ENCODER_PATH,
    LOCATION_ENCODER_PATH
)

from utils.preprocess import preprocess

# Load model & preprocessors
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
device_encoder = joblib.load(DEVICE_ENCODER_PATH)
location_encoder = joblib.load(LOCATION_ENCODER_PATH)


def predict_transaction(data):
    X = preprocess(data, scaler, device_encoder, location_encoder)
    prob = float(model.predict(X, verbose=0)[0][0])

    # Determine risk level
    if prob < 0.3:
        risk_level = "Normal"
        prediction = "SAFE"
        message = "Transaction is safe to process"
    elif 0.3 <= prob < 0.7:
        risk_level = "Suspicious"
        prediction = "SUSPICIOUS"
        message = "Medium-risk transaction detected"
    else:
        risk_level = "High Risk"
        prediction = "FRAUD"
        message = "High-risk transaction detected"

    return {
        "transaction_id": data["transaction_id"],
        "prediction": prediction,
        "risk_level": risk_level,
        "fraud_probability": round(prob, 2),
        "message": message
    }
