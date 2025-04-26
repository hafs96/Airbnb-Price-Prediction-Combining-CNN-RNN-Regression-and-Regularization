import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import os
import joblib

# Messages de progression
print("Chargement des données...")
data = pd.read_csv("C:/Users/l/Desktop/ML & IA/S6/deep learning -Monir/ZIAN HAFSA- Project/Data/Airbnb_Data.csv")

print("Remplissage des valeurs manquantes...")
data['bathrooms'] = data['bathrooms'].fillna(data['bathrooms'].median())
data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].median())
data['beds'] = data['beds'].fillna(data['beds'].median())

print("Encodage des variables catégoriques...")
data = pd.get_dummies(data, columns=["property_type", "room_type", "cancellation_policy"], drop_first=True)

print("Normalisation des données structurées...")
scaler = StandardScaler()
X_structured = scaler.fit_transform(data[["accommodates", "bathrooms", "bedrooms", "review_scores_rating"]])

print("Prétraitement des descriptions textuelles...")
vectorizer = TfidfVectorizer(max_features=1000)
description_vectors = vectorizer.fit_transform(data["description"])

# Sauvegarder les vecteurs TF-IDF
output_dir = "../Data"
os.makedirs(output_dir, exist_ok=True)
scipy.sparse.save_npz(os.path.join(output_dir, "description_vectors.npz"), description_vectors)

# Sauvegarder l'objet TfidfVectorizer
joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
print("Vecteur TF-IDF et TfidfVectorizer sauvegardés avec succès.")

print("Sauvegarde des données prétraitées...")
data.to_csv(os.path.join(output_dir, "processed_data.csv"), index=False)

print("Prétraitement terminé avec succès !")