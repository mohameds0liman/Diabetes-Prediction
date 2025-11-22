import os
from fastapi import FastAPI
from pydantic import BaseModel , Field ,validator
import joblib

current_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(current_dir,"diabetes_model.pkl")
app=FastAPI()
model=joblib.load(model_path)
print("Model Loaded!")
class user_inputs(BaseModel):
    Pregnancies: int   =Field(default=... , description="Pregnancies" , examples=[2] , ge=0, le=20)
    Glucose:int         =Field(default=...  , description="Pregnancies" , examples=[120] , ge=0, le=300)
    BloodPressure:int   =Field(default=...  , description="Pregnancies" , examples=[70] , ge=0, le=200)
    SkinThickness:int   =Field(default=...  , description="Pregnancies" , examples=[25] , ge=0, le=100)
    Insulin:int         =Field(default=...  , description="Pregnancies" , examples=[90] , ge=0, le=900)
    BMI:float             =Field(default=...  , description="Pregnancies" , examples=[30.5] , ge=0, le=70)
    DiabetesPedigreeFunction:float    =Field(default=...  , description="Pregnancies" , examples=[0.52] , ge=0, le=3)
    Age	:int            =Field(default=...  , description="Pregnancies" , examples=[33] , ge=0, le=120)
    @validator("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age", pre=True)
    def to_int(cls, v):
            return int(v)

@app.get("/Home")
async def Home():
    return {"Message" : "Welcom Home"}

@app.post("/Diabetes_Classification")
async def user(data: user_inputs):
    data_dict=data.dict()
    user_input=list(data_dict.values())
    pred=model.predict([user_input])
    if pred:
        return {"Message" : "You Have Diabetes"}
    else:
        return {"Message" : "You Do not Have Diabetes"}
    