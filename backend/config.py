import os

# Go to project root (1 level up from backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_lstm_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DEVICE_ENCODER_PATH = os.path.join(BASE_DIR, "models", "device_encoder.pkl")
LOCATION_ENCODER_PATH = os.path.join(BASE_DIR, "models", "location_encoder.pkl")
