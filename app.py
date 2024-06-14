# app.py
import os
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Step 1: Data Collection
drug_labeling_url = "https://api.fda.gov/drug/label.json"
adverse_events_url = "https://api.fda.gov/drug/event.json"

drug_labeling_data = requests.get(drug_labeling_url).json()
adverse_events_data = requests.get(adverse_events_url).json()

# Step 2: Data Preprocessing
drug_labeling_df = pd.DataFrame(drug_labeling_data["results"])
adverse_events_df = pd.DataFrame(adverse_events_data["results"])

# Data cleaning and feature engineering for drug labeling data
drug_labeling_df = drug_labeling_df.dropna(subset=["drug_composition", "drug_indication"])
onehot_encoder = OneHotEncoder(sparse=False)
drug_compositions = onehot_encoder.fit_transform(drug_labeling_df[["drug_composition"]])
drug_indications = onehot_encoder.fit_transform(drug_labeling_df[["drug_indication"]])
X_labeling = np.concatenate([drug_compositions, drug_indications], axis=1)

# Data cleaning and feature engineering for adverse events data
adverse_events_df = adverse_events_df.dropna(subset=["patient_age", "patient_sex", "adverse_event_type"])
label_encoder = LabelEncoder()
adverse_events_df["patient_sex"] = label_encoder.fit_transform(adverse_events_df["patient_sex"])
X_events = adverse_events_df[["patient_age", "patient_sex"]].values
y_events = adverse_events_df["adverse_event_type"]

# Combine features from both datasets
X = np.concatenate([X_labeling, X_events], axis=1)
y = y_events

# Step 3: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 5: Save the trained model
model_path = os.path.join("models", "adverseevent_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(model, f)
