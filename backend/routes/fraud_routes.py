from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.predict import predict_transaction

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return jsonify(predict_transaction(data))

if __name__ == "__main__":
    app.run(debug=True)
