# drug_adverse_event_model_colab.py

import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime

# Setup logging to a file named drug_adverse_event_model.log
logging.basicConfig(filename='drug_adverse_event_model.log', level=logging.INFO)

def fetch_data(limit=100):
    url = f"https://api.fda.gov/drug/event.json?limit={limit}"
    response = requests.get(url)
    data = response.json()
    events = data['results']
    
    # Extract relevant data
    records = []
    for event in events:
        record = {
            'drug': event['patient']['drug'][0]['medicinalproduct'],
            'age': event['patient'].get('patientonsetage', None),
            'sex': event['patient'].get('patientsex', None),
            'serious': event.get('serious', None),  # General serious field
            'outcome': 1 if event.get('serious', '0') == '1' else 0  # 1 if serious, else 0
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def preprocess_data(df):
    df['sex'] = df['sex'].apply(lambda x: 1 if x == '1' else 0)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
    df['serious'] = pd.to_numeric(df['serious'], errors='coerce').fillna(0)
    return df[['age', 'sex', 'serious']], df['outcome']

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

def main():
    try:
        data = fetch_data()
        X, y = preprocess_data(data)
        accuracy, report = train_model(X, y)
        logging.info(f"{datetime.now()}: Model trained with accuracy = {accuracy}")
        logging.info(f"Classification Report:\n{report}")
    except Exception as e:
        logging.error(f"{datetime.now()}: Error occurred - {e}")

if __name__ == "__main__":
    main()
