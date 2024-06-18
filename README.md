# DrugMeNot - Adverse Drug Event Prediction Model

This project aims to develop a machine learning model for predicting the seriousness of adverse events associated with drug usage based on patient information and drug characteristics. The model is trained on data obtained from the FDA Adverse Event Reporting System (FAERS) API.

## Features

- Data Collection: The project fetches data from the FDA FAERS API, handling potential errors and rate limiting.

- Data Preprocessing: The data is preprocessed to handle missing values, normalize nested JSON columns, and encode categorical features using techniques like label encoding and one-hot encoding.
- Feature Engineering: Relevant features are extracted from the raw data, including patient age, sex, drug composition, drug indication, and reaction information.
- Model Training and Evaluation: The project trains a Random Forest Classifier model on the preprocessed data and evaluates its performance using classification metrics such as precision, recall, and F1-score.
- Logging: Comprehensive logging is implemented to track the progress and errors encountered during the execution of the project.

## Requirements

Python 3.x
pandas
numpy
scikit-learn
requests

## Usage

Clone the repository:

```
git clone https://github.com/your-username/adverse-event-prediction-model.git
```

Navigate to the project directory:

```
cd adverse-event-prediction-model
```

Install the required dependencies:

```
pip install -r requirements.txt
```

Run the main script:

```
python adverse_event_model.py
```

The script will fetch data from the FDA FAERS API, preprocess the data, train the Random Forest Classifier model, and save the trained model to the models directory.
Check the console output and the adverse_event_model.log file for the classification report and any log messages.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the Apache License.

## Acknowledgments
This project was inspired by the pharmaceutical industry's need for effective adverse event prediction models. The data used in this project is sourced from the FDA Adverse Event Reporting System (FAERS) API.
