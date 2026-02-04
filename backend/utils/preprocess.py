import numpy as np

def preprocess(data, scaler, device_encoder, location_encoder):
    """
    Accepts FULL transaction JSON and extracts required fields
    """

    amount = float(data.get("amount", 0))
    time = float(data.get("time", 0))

    device = device_encoder.transform(
        [data.get("device_type", "mobile")]
    )[0]

    location = location_encoder.transform(
        [data.get("location", "Delhi")]
    )[0]

    X = np.array([[amount, device, location, time]])
    X = scaler.transform(X)

    return X.reshape(1, X.shape[1], 1)
