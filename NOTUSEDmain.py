import predictions_elsa as p
import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

with open("scaler.pkl", "rb") as f:
    pickle.load(f)

with open("model.pkl", "rb") as f:
    pickle.load(f)

class inputUser(BaseModel):
    type : str
    subtype : str
    bedroomCount : float
    bathroomCount : float
    province : str
    locality : str
    postCode : int
    habitableSurface : float
    hasAttic : str
    hasBasement : str
    hasDressingRoom : bool
    hasDiningRoom : bool
    buildingCondition : str
    buildingConstructionYear : float
    facedeCount : float
    hasLift : bool
    floodZoneType : str
    heatingType : str
    hasHeatPump : bool
    hasPhotovoltaicPanels : bool
    hasThermicPanels : bool
    kitchenType : str
    landSurface : float
    hasLivingRoom : bool
    hasGarden : bool
    gardenSurface : float
    parkingCountIndoor : float
    parkingCountOutdoor : float
    hasAirConditioning : bool
    hasArmoredDoor : bool
    hasVisiophone : bool
    hasOffice : bool
    toiletCount : float
    hasSwimmingPool : bool
    hasFireplace : bool
    hasTerrace : bool
    terraceSurface : float
    epcScore : str

@app.post('/predict')
def user_input_preparation(user : inputUser):
    #id : float
    #url : str
    #roomCount : float
    #monthlyCost : float
    #diningRoomSurface : float
    #terraceOrientation : str
    #accessibleDisabledPeople : bool
    #floorCount : float
    #streetFacadeWidth : float
    #kitchenSurface : float
    #livingRoomSurface : float
    #hasBalcony : bool
    #gardenOrientation : str
    #price : float
    
    df_new = pd.DataFrame([user_dict])

    # 2. Nettoyage & feature engineering
    df_new_clean = p.cleaning_dataframe(df_new, is_training=False)

    # 3. Imputations
    df_new_clean_imputed = p.transform_cleaning_traintestsplit(df_new_clean, p.stats_from_X_train, is_training=False)

    # 4. Scaling (en gardant le fit de tes données d’entraînement)
    columns_for_model = p.scaler.feature_names_in_.tolist()
    df_new_clean_imputed = df_new_clean_imputed[columns_for_model]

    df_new_scaled = p.scaler.transform(df_new_clean_imputed)

    # 5. Prédiction
    predicted_price = p.xgb_model.predict(df_new_scaled)

    return (f"Le prix prédit est {predicted_price[0]:,.0f} €".replace(',', ' '))
