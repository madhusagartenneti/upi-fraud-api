import pandas as pd
import random
from datetime import datetime, timedelta

NUM_RECORDS = 25000
FRAUD_PERCENT = 0.1

locations = ["Hyderabad", "Delhi", "Mumbai", "Bangalore", "Chennai"]
devices = ["mobile", "desktop", "tablet", "emulator"]

data = []
start_time = datetime.now() - timedelta(days=180)

for i in range(NUM_RECORDS):
    fraud = 1 if random.random() < FRAUD_PERCENT else 0

    if fraud:
        amount = random.randint(30000, 120000)
        hour = random.choice([0, 1, 2, 3, 4, 23])
        device = random.choice(["emulator", "desktop"])
    else:
        amount = random.randint(10, 20000)
        hour = random.randint(6, 22)
        device = random.choice(["mobile", "tablet", "desktop"])

    data.append({
        "transaction_id": f"TXN{i}",
        "sender_upi": f"user{i}@upi",
        "receiver_upi": f"merchant{i}@upi",
        "location": random.choice(locations),
        "amount": amount,
        "device_type": device,
        "time": hour,
        "label": fraud
    })

df = pd.DataFrame(data)

# ✅ FIXED PATH
df.to_csv("upi_transactions.csv", index=False)

print("✅ Dataset created successfully")
print(df["label"].value_counts())
