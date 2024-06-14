import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
from datetime import datetime
import re
import numpy as np

# Configure logging to write to a file
logging.basicConfig(filename='drug_adverse_event_model.log', level=logging.INFO)

def fetch_data(limit=100):
    try:
        url = f"https://api.fda.gov/drug/event.json?limit={limit}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad response status
        data = response.json()
        events = data.get('results', [])

        # Extract relevant data
        records = []
        for event in events:
            drug_info = event.get('patient', {}).get('drug', [])[0]
            record = {
                'drug': drug_info.get('medicinalproduct', None),
                'age': event.get('patient', {}).get('patientonsetage', None),
                'sex': event.get('patient', {}).get('patientsex', None),
                'dose': drug_info.get('drugdosagetext', None),  
                'indication': drug_info.get('drugindication', None),
                'route': drug_info.get('drugadministrationroute', None),
                'duration': event.get('receivedate', None),  
                'outcome': 1 if event.get('serious', '0') == '1' else 0  
            }
            records.append(record)

        df = pd.DataFrame(records)
        return df

    except Exception as e:
        logging.error(f"{datetime.now()}: Error fetching data - {str(e)}")
        raise  # Re-raise the exception for higher-level handling

def extract_numeric_value(value):
    if pd.isna(value):
        return np.nan
    
    try:
        match = re.search(r'\d+(\.\d+)?', str(value))
        if match:
            return float(match.group())
        else:
            return np.nan
    except ValueError:
        return np.nan

def preprocess_data(df):
    try:
        if 'outcome' not in df.columns:
            raise ValueError("DataFrame must include 'outcome' column")
        
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['sex'] = df['sex'].apply(lambda x: 1 if x == '1' else 0)
        df['dose'] = df['dose'].apply(extract_numeric_value)
        df['duration'] = pd.to_datetime(df['duration'], format='%Y-%m-%d', errors='coerce')
        df['duration'] = (datetime.now() - df['duration']).dt.days
        
        X = df.drop(columns=['outcome'])
        y = df['outcome']
        
        num_features = ['age', 'dose', 'duration']
        cat_features = ['drug', 'indication', 'route']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ], remainder='passthrough'
        )
        
        X_transformed = preprocessor.fit_transform(X)
        
        imputer = SimpleImputer(strategy='median')
        X_transformed = imputer.fit_transform(X_transformed)
        
        return X_transformed, y, preprocessor, imputer
    
    except Exception as e:
        logging.error(f"{datetime.now()}: Error preprocessing data - {str(e)}")
        raise  # Re-raise the exception for higher-level handling

def main():
    try:
        logging.info(f"{datetime.now()}: Starting data fetch...")
        data = fetch_data()
        logging.info(f"{datetime.now()}: Data fetched successfully.")

        logging.info(f"{datetime.now()}: Splitting data into train and test sets...")
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        logging.info(f"{datetime.now()}: Preprocessing training data...")
        X_train, y_train, preprocessor, imputer = preprocess_data(train_data)

        logging.info(f"{datetime.now()}: Preprocessing test data...")
        X_test = preprocessor.transform(test_data.drop(columns=['outcome']))
        X_test = imputer.transform(X_test)
        y_test = test_data['outcome']
        
        logging.info(f"{datetime.now()}: Data preprocessing complete. Number of features: {X_train.shape[1]}")

        assert X_train.shape[1] == X_test.shape[1], \
            f"Training and testing data have different number of features: {X_train.shape[1]} vs {X_test.shape[1]}"

        logging.info(f"{datetime.now()}: Training model...")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logging.info(f"{datetime.now()}: Model trained with accuracy = {accuracy}")
        logging.info(f"Classification Report:\n{report}")

    except Exception as e:
        logging.error(f"{datetime.now()}: Error occurred - {str(e)}")

if __name__ == "__main__":
    main()
