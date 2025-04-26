import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import numpy as np
import joblib

# Vérification du répertoire courant
print("Répertoire courant :", os.getcwd())

# Changez le répertoire courant pour qu'il corresponde à la racine du projet (si nécessaire)
project_root = "C:/Users/l/Desktop/ML & IA/S6/deep learning -Monir/ZIAN HAFSA- Project"
os.chdir(project_root)
print("Répertoire courant après changement :", os.getcwd())

# Charger les données prétraitées
try:
    data_path = "Data/processed_data.csv"
    print(f"Chargement des données depuis : {data_path}")
    data = pd.read_csv(data_path, low_memory=False)  # Ajout de low_memory=False pour éviter DtypeWarning
except FileNotFoundError:
    print(f"Erreur : Le fichier '{data_path}' est introuvable.")
    exit(1)

# Charger les vecteurs TF-IDF
try:
    tfidf_path = "Data/description_vectors.npz"
    print(f"Chargement des vecteurs TF-IDF depuis : {tfidf_path}")
    description_vectors = scipy.sparse.load_npz(tfidf_path)
except FileNotFoundError:
    print(f"Erreur : Le fichier '{tfidf_path}' est introuvable.")
    exit(1)

# Charger l'objet TfidfVectorizer
try:
    vectorizer_path = "Data/tfidf_vectorizer.joblib"
    print(f"Chargement de TfidfVectorizer depuis : {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    print(f"Erreur : Le fichier '{vectorizer_path}' est introuvable.")
    exit(1)

# Caractéristiques structurées
print("Sélection des caractéristiques...")
X_structured = data[["accommodates", "bathrooms", "bedrooms", "review_scores_rating"]]

# Combinaison des caractéristiques structurées et textuelles
X_combined = scipy.sparse.hstack([X_structured, description_vectors])

# Variable cible
y = data["log_price"]

# Diviser les données en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Remplir les valeurs manquantes avec la médiane
print("Traitement des valeurs manquantes...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Initialiser et entraîner le modèle
print("Entraînement de la régression linéaire...")
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions
print("Évaluation du modèle...")
y_pred = model.predict(X_test)

# Calculer l'erreur RMSE
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    print("Calcul de RMSE manuellement (version de Scikit-learn obsolète)...")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
print(f"RMSE : {rmse}")

# Interpréter les coefficients
feature_names = ["accommodates", "bathrooms", "bedrooms", "review_scores_rating"]
coefficients = dict(zip(feature_names, model.coef_[:len(feature_names)]))
print("\nCoefficients des caractéristiques structurées :")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef:.4f}")

# Afficher les mots les plus influents dans les descriptions
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_coefficients = model.coef_[len(feature_names):]
top_words = sorted(zip(tfidf_feature_names, tfidf_coefficients), key=lambda x: abs(x[1]), reverse=True)[:10]
print("\nMots les plus influents dans les descriptions :")
for word, coef in top_words:
    print(f"{word}: {coef:.4f}")

# Visualiser les résultats
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Parfaite Prédiction")
plt.xlabel("True Prices (log scale)")
plt.ylabel("Predicted Prices (log scale)")
plt.title("True vs Predicted Prices")
plt.legend()
plt.show()

print("Entraînement terminé avec succès !")