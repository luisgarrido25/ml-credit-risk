from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Obtener ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ruta del modelo
model_path = os.path.join(BASE_DIR, "models", "credit_model_v1.joblib")

# Cargar modelo
model = joblib.load(model_path)


@app.route("/", methods=["GET"])
def health():
    return {"status": "API funcionando"}


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()
        instances = data.get("instances", data)
        df = pd.DataFrame(instances)

        predictions = model.predict(df)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# logging simple
print("Request received")