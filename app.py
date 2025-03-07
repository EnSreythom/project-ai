from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the input data schema
class InputData(BaseModel):
    gender: int
    age: int
    hypertension: int
    heart_disease: int
    ever_married: int
    work_type: int
    Residence_type: int
    avg_glucose_level: float
    bmi: float
    smoking_status: int

# Initialize FastAPI app
app = FastAPI()

# Serve static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())

# API endpoint for predictions
@app.post("/predict")
async def predict(data: InputData):
    # Convert input data to numpy array
    input_data = np.array([
        data.gender, data.age, data.hypertension, data.heart_disease,
        data.ever_married, data.work_type, data.Residence_type,
        data.avg_glucose_level, data.bmi, data.smoking_status
    ]).reshape(1, -1)
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Get prediction probabilities (if the model supports it)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data_scaled)[0]
        probability = probabilities[1]  # Probability of class 1 (stroke)
    else:
        probability = None  # If the model doesn't support probabilities

    # Return the prediction and probability
    return JSONResponse(content={
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else None
    })

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)