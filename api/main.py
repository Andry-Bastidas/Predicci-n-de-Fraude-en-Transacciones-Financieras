import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sqlmodel import Session , select
from enum import Enum
from .app.model import ProcessDataRequestModel 
from .app.db import engine , PredictionsTransaccion , create_db_and_tables
from .app.utils import dataframe_transform  , transform_data , scaler_x




app = FastAPI(title= "FastAPI , Docker and grafana")

label_mapping = {
    0 : 'No_Fradulento',
    1 : 'Is_Fradulento'
}

@app.post("/predict")
async def predictions(data:ProcessDataRequestModel):
    
    session = Session(engine)

    model = joblib.load('Model/model.pkl')

    pred_list = []

    for trans in data.transacciones:

        datos = vars(trans)
        dataframe = dataframe_transform(datos)
        process_data = transform_data(dataframe)
        process_data_scaler = scaler_x(process_data)

        preds = model.predict(process_data_scaler)
        decoded_predictions = label_mapping[preds[0]]

        predictions_trans = PredictionsTransaccion(
            client_name = trans.first ,
            client_last_name = trans.last,
            client_cc_num = str(trans.cc_num),
            trans_time = trans.trans_date_trans_time,
            prediction = decoded_predictions
        )

        print(predictions_trans)

        pred_list.append(
            {
                "client_name" :  trans.first,
                "client_last_name" : trans.last,
                "client_cc_num" : str(trans.cc_num),
                "trans_time" : trans.trans_date_trans_time,
                'predictions' : decoded_predictions
            }
        )

        session.add(predictions_trans)

    session.commit()
    session.close()

    return {"predictions" : pred_list}

@app.on_event("startup")
async def starup():
    create_db_and_tables()

