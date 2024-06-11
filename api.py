import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model_reloaded = joblib.load("mnist.pkl")

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    rgb_number = np.array(data)
    predictions = final_model_reloaded.predict([rgb_number])
    return jsonify({"prediction": int(predictions[0])})


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8081)
