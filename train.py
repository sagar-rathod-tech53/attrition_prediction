import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

X_train = pd.read_csv(os.path.join(DATASET_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET_DIR, "y_train.csv"))
y_test = pd.read_csv(os.path.join(DATASET_DIR, "y_test.csv"))

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(f"{name}: {acc:.4f}")

# Best model
best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])

# Save models
os.makedirs("../models", exist_ok=True)

joblib.dump(models["Logistic Regression"],
            "../models/logistic_regression.pkl")

joblib.dump(models["Decision Tree"],
            "../models/decision_tree.pkl")

joblib.dump(models["Random Forest"],
            "../models/random_forest.pkl")

joblib.dump(models["KNN"],
            "../models/knn.pkl")

joblib.dump(scaler, "../models/scaler.pkl")

# Save column names
joblib.dump(list(X_train.columns),
            "../models/columns.pkl")

print("All models saved successfully!")