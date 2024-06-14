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
drug_labeling_results = drug_labeling_data.get("results", [])
adverse_events_results = adverse_events_data.get("results", [])

drug_labeling_df = pd.json_normalize(drug_labeling_results, record_path=['openfda', 'drug_composition'])
adverse_events_df = pd.json_normalize(adverse_events_results, record_path=['patient'])

# Data cleaning and feature engineering for drug labeling data
drug_compositions = drug_labeling_df["name"].tolist()
onehot_encoder = OneHotEncoder(sparse=False)
X_labeling = onehot_encoder.fit_transform(np.array(drug_compositions).reshape(-1, 1))

# Data cleaning and feature engineering for adverse events data
adverse_events_df = adverse_events_df.dropna(subset=["patientonsetage", "patientsex", "reaction"])
label_encoder = LabelEncoder()
adverse_events_df["patientsex"] = label_encoder.fit_transform(adverse_events_df["patientsex"])
adverse_events_df["reaction"] = adverse_events_df["reaction"].apply(lambda x: [d["reactionmeddrapt"] for d in x])
X_events = adverse_events_df[["patientonsetage", "patientsex"]].values
y_events = adverse_events_df["reaction"].apply(lambda x: ','.join(x)).values

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
