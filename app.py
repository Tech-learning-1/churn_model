from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/churn_model.pkl")

class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges:float
    Total_charges: float
    support_calls: int


@app.get("/health")
def health():
    return {"status": "ok"}
    
@app.post("/predict")
def predict(data:CustomerData):
    features = np.array([[data.age,data.tenure_months,data.monthly_charges,data.Total_charges,data.support_calls]])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    return {

        "churn" : int(pred),
        "probability" : float(proba)
    }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
