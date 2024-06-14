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
    url = f"https://api.fda.gov/drug/event.json?limit={limit}"
    response = requests.get(url)
    data = response.json()
    events = data['results']
    
    # Extract relevant data
    records = []
    for event in events:
        drug_info = event['patient']['drug'][0]
        record = {
            'drug': drug_info['medicinalproduct'],
            'age': event['patient'].get('patientonsetage', None),
            'sex': event['patient'].get('patientsex', None),
            'dose': drug_info.get('drugdosagetext', None),  # Assuming dose information is in the 'drugdosagetext' field
            'indication': drug_info.get('drugindication', None),
            'route': drug_info.get('drugadministrationroute', None),
            'duration': event.get('drugstartdate', None),  # Assuming duration information is in the 'drugstartdate' field
            'outcome': 1 if event.get('serious', '0') == '1' else 0  # 1 if serious, else 0
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df

def extract_numeric_value(value):
    if pd.isna(value):
        return np.nan
    
    try:
        # Use regular expression to find the first sequence of digits or decimals in the string
        match = re.search(r'\d+(\.\d+)?', str(value))
        if match:
            return float(match.group())
        else:
            return np.nan
    except ValueError:
        return np.nan

def preprocess_data(df):
    # Ensure the outcome column is included
    if 'outcome' not in df.columns:
        raise ValueError("DataFrame must include 'outcome' column")
    
    # Convert 'age' column to numeric, handling errors with 'coerce'
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Convert 'sex' column to binary (1 for male, 0 for female)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == '1' else 0)
    
    # Extract numerical values from 'dose' column using custom function
    df['dose'] = df['dose'].apply(extract_numeric_value)
    
    # Convert 'duration' to numeric (if it is a date string, we need to convert it to duration in days or similar)
    # Assuming 'duration' is a date string in the format 'YYYYMMDD'
    df['duration'] = pd.to_datetime(df['duration'], format='%Y%m%d', errors='coerce')
    df['duration'] = (datetime.now() - df['duration']).dt.days
    
    # Separate features and target
    X = df.drop(columns=['outcome'])
    y = df['outcome']
    
    # Create a column transformer with separate transformers
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
    
    # Fit on all data and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_transformed = imputer.fit_transform(X_transformed)
    
    return X_transformed, y, preprocessor, imputer

def main():
    try:
        logging.info(f"{datetime.now()}: Starting data fetch...")
        data = fetch_data()
        logging.info(f"{datetime.now()}: Data fetched successfully.")

        # Ensure 'outcome' column is in the dataframe
        if 'outcome' not in data.columns:
            logging.error(f"{datetime.now()}: 'outcome' column is missing from data")
            return

        # Perform train-test split
        logging.info(f"{datetime.now()}: Splitting data into train and test sets...")
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

        logging.info(f"{datetime.now()}: Preprocessing data...")
        X_train, y_train, preprocessor, imputer = preprocess_data(train_data)
        X_test = preprocessor.transform(test_data.drop(columns=['outcome']))
        X_test = imputer.transform(X_test)
        y_test = test_data['outcome']
        
        logging.info(f"{datetime.now()}: Data preprocessing complete. Number of features: {X_train.shape[1]}")

        # Ensure that both X_train and X_test have the same shape
        assert X_train.shape[1] == X_test.shape[1], \
            f"Training and testing data have different number of features: {X_train.shape[1]} vs {X_test.shape[1]}"

        logging.info(f"{datetime.now()}: Training model...")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Predict on test set
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        logging.info(f"{datetime.now()}: Model trained with accuracy = {accuracy}")
        logging.info(f"Classification Report:\n{report}")

    except Exception as e:
        logging.error(f"{datetime.now()}: Error occurred - {e}")

if __name__ == "__main__":
    main()
