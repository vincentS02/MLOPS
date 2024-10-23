from typing import Union

from fastapi import FastAPI
from model_utils import model_predict
import logging
from fastapi import HTTPException
from fastapi.logger import logger
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{size}/{nb_rooms}/{garden}")
def predict(size: float, nb_rooms: float, garden: str):
    try:
        logger.info(f"Received prediction request with size={size}, nb_rooms={nb_rooms}, garden={garden}")
        prediction = model_predict(size, nb_rooms, garden)
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

