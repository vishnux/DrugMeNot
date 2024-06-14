import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime

# Configure logging to write to a file
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
    # Convert 'age' column to numeric, handling errors with 'coerce'
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Fill NaN values with median age
    median_age = df['age'].median()
    df['age'].fillna(median_age, inplace=True)
    
    # Convert 'serious' column to numeric, handling errors with 'coerce'
    df['serious'] = pd.to_numeric(df['serious'], errors='coerce')
    
    # Fill NaN values with 0 (or appropriate handling based on your data)
    df['serious'].fillna(0, inplace=True)
    
    # Convert 'sex' column to binary (if needed)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == '1' else 0)
    
    return df[['age', 'sex', 'serious']], df['outcome']

def main():
    try:
        logging.info(f"{datetime.now()}: Starting data fetch...")
        data = fetch_data()
        logging.info(f"{datetime.now()}: Data fetched successfully.")

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(data[['age', 'sex', 'serious']], 
                                                            data['outcome'], 
                                                            test_size=0.3, 
                                                            random_state=42)

        logging.info(f"{datetime.now()}: Preprocessing training data...")
        X_train_processed, y_train_processed = preprocess_data(X_train)
        logging.info(f"{datetime.now()}: Training data preprocessing complete.")

        logging.info(f"{datetime.now()}: Preprocessing testing data...")
        X_test_processed, y_test_processed = preprocess_data(X_test)
        logging.info(f"{datetime.now()}: Testing data preprocessing complete.")

        logging.info(f"{datetime.now()}: Training model...")
        model = RandomForestClassifier()
        model.fit(X_train_processed, y_train_processed)
        
        # Predict on test set
        predictions = model.predict(X_test_processed)
        
        accuracy = accuracy_score(y_test_processed, predictions)
        report = classification_report(y_test_processed, predictions)
        logging.info(f"{datetime.now()}: Model trained with accuracy = {accuracy}")
        logging.info(f"Classification Report:\n{report}")

    except Exception as e:
        logging.error(f"{datetime.now()}: Error occurred - {e}")

if __name__ == "__main__":
    main()
