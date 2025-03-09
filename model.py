import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Load the processed dataset
df = pd.read_csv('processed_fraud_dataset.csv')

# Feature Engineering
X = df.drop('Fraud', axis=1)
y = df['Fraud']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Encoding categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols]).toarray()

# Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])

# Combine encoded and scaled features
X_processed = np.hstack([X_scaled, X_encoded])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation
y_pred = rf_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))

# Save model and preprocessing artifacts
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(rf_model, 'model_artifacts/fraud_detection_model.joblib')
joblib.dump(encoder, 'model_artifacts/encoder.joblib')
joblib.dump(scaler, 'model_artifacts/scaler.joblib')

print("Model and preprocessing steps have been saved successfully!")
