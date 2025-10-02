import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

os.makedirs("models", exist_ok=True)

# create synthetic data
n = 3000
rng = np.random.RandomState(42)
age = rng.randint(20, 90, size=n)
bp_systolic = rng.normal(130, 15, n)
hr = rng.normal(75, 10, n)
creatinine = rng.normal(1.0, 3.0, n)
hemoglobin = rng.normal(13.5, 1.5, n)
diabetes = rng.binomial(1, 0.22, n)

# Probit / logistic function that defines "true" risk probability
logit = 0.03*(age - 50) + 0.02*(bp_systolic - 120) + 0.9*diabetes + 0.6*(creatinine - 1.0)
prob = 1.0 / (1.0 + np.exp(-logit))
y = (prob > 0.5).astype(int)

# feature matrix
X = np.vstack([age, bp_systolic, hr, creatinine, hemoglobin, diabetes]).T

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
clf = LogisticRegression(max_iter=1000).fit(X_scaled, y)

joblib.dump({"scaler": scaler, "clf": clf}, "models/risk_model.joblib")
print("Saved risk model to models/risk_model.joblib")