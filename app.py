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

# Check for common columns to merge on
common_columns = set(drug_labeling_df.columns).intersection(set(adverse_events_df.columns))
merge_on_column = None
for col in ["product_ndc", "product_code"]:
    if col in common_columns:
        merge_on_column = col
        break

if merge_on_column is None:
    raise ValueError("No common columns found to merge the dataframes.")

# Filter and join relevant data
relevant_data = pd.merge(drug_labeling_df, adverse_events_df, on=merge_on_column, how="inner")

# Data cleaning
# Handle missing values
relevant_data = relevant_data.dropna(subset=["patient_age", "patient_sex", "drug_composition", "drug_indication", "adverse_event_type"])

# Feature engineering
# Encode categorical variables
label_encoder = LabelEncoder()
relevant_data["patient_sex"] = label_encoder.fit_transform(relevant_data["patient_sex"])

onehot_encoder = OneHotEncoder(sparse=False)
drug_compositions = onehot_encoder.fit_transform(relevant_data[["drug_composition"]])
drug_indications = onehot_encoder.fit_transform(relevant_data[["drug_indication"]])

# Combine features
X = np.concatenate([relevant_data[["patient_age"]].values, relevant_data[["patient_sex"]].values, drug_compositions, drug_indications], axis=1)
y = relevant_data["adverse_event_type"]

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
