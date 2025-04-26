# src/predict.py
import pandas as pd
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model("../models/trained_model.h5")

# Charger les données de test
test_data = pd.read_csv("../Data/test_data.csv")

# Faire des prédictions
predictions = model.predict(test_data)
print(predictions)