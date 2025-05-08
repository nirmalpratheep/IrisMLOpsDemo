
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
model = joblib.load("iris_model.pkl")

instrumentator = Instrumentator().instrument(app).expose(app)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(req: IrisRequest):
    data = np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]])
    pred = model.predict(data)[0]
    return {"prediction": pred}
