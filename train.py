# train.py
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

np.random.seed(42)
n_samples = 1000

# Features: [Age, BMI, BloodPressure, Glucose/Cholesterol, FamilyHistory]
age = np.random.uniform(18, 85, n_samples)
bmi = np.random.uniform(16, 40, n_samples)
bp = np.random.uniform(90, 180, n_samples)
glucose_chol = np.random.uniform(60, 260, n_samples)
family_hist = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

X_train = np.stack([age, bmi, bp, glucose_chol, family_hist], axis=1)

# Helper function to generate labels based on risk rules + noise
def generate_labels(prob_formula):
    probs = prob_formula(X_train)
    probs = np.clip(probs, 0.05, 0.95)
    return np.random.binomial(1, probs)

# Diabetes risk: heavily influenced by glucose, BMI, age, and family history
y_diabetes = generate_labels(lambda X: (
    0.4 * ((X[:, 3] - 60) / 200) +  # glucose
    0.25 * ((X[:, 1] - 16) / 24) +  # BMI
    0.15 * ((X[:, 0] - 18) / 67) +  # age
    0.2 * X[:, 4]                   # family history
))

# Heart Disease risk: influenced by cholesterol, BP, age, and family history
y_heart = generate_labels(lambda X: (
    0.35 * ((X[:, 3] - 60) / 200) +  # cholesterol
    0.3 * ((X[:, 2] - 90) / 90) +    # BP
    0.2 * ((X[:, 0] - 18) / 67) +   # age
    0.15 * X[:, 4]                  # family history
))

# Hypertension risk: heavily influenced by BP, BMI, and age
y_hypertension = generate_labels(lambda X: (
    0.55 * ((X[:, 2] - 90) / 90) +   # BP
    0.25 * ((X[:, 1] - 16) / 24) +   # BMI
    0.2 * ((X[:, 0] - 18) / 67)      # age
))

# Kidney Disease risk: influenced by BP, glucose, age
y_kidney = generate_labels(lambda X: (
    0.35 * ((X[:, 2] - 90) / 90) +      # BP
    0.35 * ((X[:, 3] - 60) / 200) +     # glucose/lab
    0.2 * ((X[:, 0] - 18) / 67) +       # age
    0.1 * X[:, 4]                       # family history
))

# Train and save models
models = {
    "diabetes": y_diabetes,
    "heart": y_heart,
    "hypertension": y_hypertension,
    "kidney": y_kidney
}

for name, y in models.items():
    model = HistGradientBoostingClassifier(max_iter=50, random_state=42)
    model.fit(X_train, y)
    joblib.dump(model, f"{name}_model.pkl")
    print(f"Model trained and saved: {name}_model.pkl")

print("All models successfully trained and saved!")