from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

# Add the model directory to the sys.path
model_path = Path(__file__).resolve().parent.parent / "model"
sys.path.append(str(model_path))

import model as model_utils

# Load the model
model = model_utils.load_model()

app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisData):
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
