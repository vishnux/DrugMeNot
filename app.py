import os
import pickle
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

# Configure logging
logging.basicConfig(filename='adverse_event_model.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Step 1: Data Collection
def fetch_data(url, limit=100):
    try:
        response = requests.get(f"{url}?limit={limit}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

adverse_events_url = "https://api.fda.gov/drug/event.json"#?api_key=8DcjSRvomd4ddOi4zRzVKqX4StK5TUv3cIl0v7OB
adverse_events_data = fetch_data(adverse_events_url)

if not adverse_events_data:
    logging.error("Failed to fetch data, terminating program.")
    exit()

# Step 2: Data Preprocessing
def preprocess_data(data):
    try:
        results = data.get("results", [])
        df = pd.DataFrame(results)

        # Filter out rows with missing patient data
        df = df[df["patient"].notnull()]

        # Normalize nested JSON columns
        patient_data = pd.json_normalize(df["patient"])
        df = df.join(patient_data)
        
        # Drop rows with critical missing values
        df = df.dropna(subset=["patientonsetage", "patientsex", "reaction", "drug"])

        # Filter out rows with empty lists in 'reaction'
        df = df[df["reaction"].apply(len) > 0]

        # Encode 'patientsex'
        label_encoder = LabelEncoder()
        df["patientsex"] = label_encoder.fit_transform(df["patientsex"])

        # Extract and encode reactions and drug information
        df["reaction"] = df["reaction"].apply(lambda x: [d["reactionmeddrapt"] for d in x])

        # Extract drug information
        def extract_drug_info(drug_list, key):
            return [d[key] for d in drug_list if key in d]

        df["drug_composition"] = df["drug"].apply(lambda x: extract_drug_info(x, "medicinalproduct"))
        df["drug_indication"] = df["drug"].apply(lambda x: extract_drug_info(x, "drugindication"))

        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise


adverse_events_df = preprocess_data(adverse_events_data)

# Prepare features and labels
def prepare_features(df):
    try:
        onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Handle drug_composition
        df['drug_composition_str'] = df['drug_composition'].apply(lambda x: ','.join(x))
        df['drug_indication_str'] = df['drug_indication'].apply(lambda x: ','.join(x))

        compositions_encoded = onehot_encoder.fit_transform(df[['drug_composition_str']])
        indications_encoded = onehot_encoder.fit_transform(df[['drug_indication_str']])

        # Combine features into a single array
        X = np.hstack([
            compositions_encoded,
            indications_encoded,
            df[["patientonsetage", "patientsex"]].values
        ])

        # Create labels
        y = df["serious"].values  # Assuming 'serious' is the column indicating seriousness

        return X, y
    except Exception as e:
        logging.error(f"Error preparing features: {e}")
        raise

X, y = prepare_features(adverse_events_df)

# Filter out rows with empty labels
non_empty_mask = y != ""
X = X[non_empty_mask]
y = y[non_empty_mask]

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Model Training and Evaluation
def train_and_evaluate(X, y, model, model_name):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, zero_division=0)
        print(f"Classification report for {model_name}:\n{report}")
        logging.info(f"Model training completed for {model_name}. Classification report:\n{report}")

        return model
    except Exception as e:
        logging.error(f"Error during model training or evaluation for {model_name}: {e}")
        raise

# Train and evaluate RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model = train_and_evaluate(X, y, rf_model, "RandomForestClassifier")

# Step 4: Save the trained model
def save_model(model, model_name):
    try:
        model_path = os.path.join("models", f"{model_name}_model.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model {model_name}: {e}")
        raise

save_model(rf_model, "RandomForest")
