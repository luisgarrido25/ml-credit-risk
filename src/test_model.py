import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "credit_model_v1.joblib")

model = joblib.load(model_path)

print("Modelo cargado correctamente")
print(type(model))