from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

# Load saved model, scaler, and encoder
model = joblib.load("fraud_detection_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class TransactionData(BaseModel):
    transactionAmount: float
    No_Transactions: int
    No_Orders: int
    No_Payments: int
    paymentMethodType: str
    paymentMethodProvider: str
    orderState: str

# Data preprocessing function
def preprocess_input(data: TransactionData):
    df = pd.DataFrame([data.dict()])
    # Encode categorical features
    encoded_df = pd.DataFrame(encoder.transform(df[['paymentMethodType', 'paymentMethodProvider', 'orderState']]))
    # Drop original categorical columns and merge encoded data
    df.drop(['paymentMethodType', 'paymentMethodProvider', 'orderState'], axis=1, inplace=True)
    processed_data = pd.concat([df, encoded_df], axis=1)
    # Scale data
    scaled_data = scaler.transform(processed_data)
    return scaled_data

# Prediction endpoint
@app.post("/predict")
async def predict(data: TransactionData):
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        return {"Fraud Prediction": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
