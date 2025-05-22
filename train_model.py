import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("dataset.csv")  # Ensure this file is in the same folder

# Display basic info
print("Dataset Preview:")
print(df.head())

# Create directory for encoders if not exists
encoder_dir = os.path.dirname(os.path.abspath(__file__))  # Save in current folder

# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

    # Save each encoder as {column_name}_encoder.pkl
    encoder_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
    joblib.dump(le, encoder_path)
    print(f"Encoder for '{col}' saved at: {encoder_path}")

# Features & Target
X = df.drop("Recurred", axis=1)
y = df["Recurred"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model to file
model_path = os.path.join(encoder_dir, "thyroid_cancer_model.pkl")
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
