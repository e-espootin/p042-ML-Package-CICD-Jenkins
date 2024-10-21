from fastapi import FastAPI
from pydantic import BaseModel  
import uvicorn
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# # Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
# # Importing the required modules
from prediction_model.predict import generate_predictions

app = FastAPI(
    title="Titanic ML CI/CD Jenkins",
    description="This is a simple ML CI/CD Jenkins",
    version="1.0"
)

# load the model
# classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TitanicPrediction(BaseModel):
    Pclass: int
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Cabin: str
    Fare: float
    Embarked: str


@app.get("/")
def index():
    return {"message": "wellcome to the Titanic ML CI/CD Jenkins"}

@app.post("/prediction_api")
def predict(passenger: TitanicPrediction):
    data = passenger.model_dump() # dict()
    # create dataframe
    df = pd.DataFrame([data])
    # predic
    print("debug data", data)
    prediction = generate_predictions(input_data=df)['prediction']
    print("debug prediction", prediction)
    # result
    if prediction == 'N':
        pred_result = "Not Survived"
    elif prediction == 'Y':
        pred_result = "Survived"
    else:
        pred_result = "Error"
    
    return {"status":pred_result}

@app.post("/prediction_ui")
def predict_gui(Pclass: int,
    Sex: str,
    Age: int,
    SibSp: int,
    Parch: int,
    Cabin: str,
    Fare: float,
    Embarked: str):
    
    data = {
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Cabin": Cabin,
        "Fare": Fare,
        "Embarked": Embarked
    }
    # create dataframe
    df = pd.DataFrame([data])
    # predic
    print("debug data", data)
    prediction = generate_predictions(input_data=df)['prediction']
    print("debug pred", prediction)
    # result
    if prediction == 'N':
        pred_result = "Not Survived"
    elif prediction == 'Y': 
        pred_result = "Survived"
    else:
        pred_result = "Error"
    
    return {"status":pred_result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
