# Example: training/train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input

# Load dataset
df = pd.read_csv("../dataset/upi_transactions.csv")
df["time"] = df["time"].astype(int)

# Encode categorical features
device_encoder = LabelEncoder()
location_encoder = LabelEncoder()
df["device_type"] = device_encoder.fit_transform(df["device_type"])
df["location"] = location_encoder.fit_transform(df["location"])

# Features and target
X = df[["amount", "device_type", "location", "time"]]
y = df["label"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle imbalance
X, y = SMOTE(random_state=42).fit_resample(X, y)

# Reshape for CNN-LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN-LSTM model
model = Sequential([
    Input(shape=(X.shape[1], 1)),
    Conv1D(64, kernel_size=2, activation="relu"),
    MaxPooling1D(2),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save("../models/cnn_lstm_model.keras")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(device_encoder, "../models/device_encoder.pkl")
joblib.dump(location_encoder, "../models/location_encoder.pkl")
print("✅ Model and preprocessors saved successfully")