from fastapi import FastAPI, HTTPException
from model2_utils import model2_predict
import logging
from fastapi.logger import logger

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Model 2 API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/predict/{feature1}/{feature2}")
def predict(feature1: float, feature2: float):
    try:
        logger.info(f"Received prediction request with feature1={feature1}, feature2={feature2}")
        prediction = model2_predict(feature1, feature2)
        logger.info(f"Prediction result: {prediction}")
        return {
            "prediction": prediction,
            "feature1": feature1,
            "feature2": feature2
        }
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3277)
