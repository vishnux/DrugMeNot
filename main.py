from fastapi import FastAPI, Query
import pickle
import numpy as np

# Load the trained model
model_path = "models/RandomForest_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# FastAPI app instance
app = FastAPI()

# Prediction endpoint
@app.get("/predict/")
def predict_adverse_event(
    patientonsetage: float = Query(..., description="Patient's age at onset of adverse event"),
    patientsex: int = Query(..., description="Patient's sex (encoded: 0 for male, 1 for female)"),
    drug_composition: str = Query(..., description="Comma-separated list of drug compositions"),
    drug_indication: str = Query(..., description="Comma-separated list of drug indications")
):
    try:
        # Prepare input features
        drug_composition_list = drug_composition.split(',')
        drug_indication_list = drug_indication.split(',')

        # One-hot encode drug compositions and indications
        onehot_encoder = model.named_steps['onehotencoder']
        compositions_encoded = onehot_encoder.transform([[','.join(drug_composition_list)]])
        indications_encoded = onehot_encoder.transform([[','.join(drug_indication_list)]])

        # Create input array
        X = np.hstack([
            compositions_encoded.toarray(),
            indications_encoded.toarray(),
            [[patientonsetage, patientsex]]
        ])

        # Make prediction
        prediction = model.predict(X)[0]

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
