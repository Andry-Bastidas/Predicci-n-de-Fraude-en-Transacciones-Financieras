from pydantic import BaseModel

class predict_model(BaseModel):
    clase : int

class Transacciones(BaseModel):
    trans_date_trans_time : str
    cc_num: int
    merchant: str
    category : str
    amt : float
    first : str
    last : str
    gender : str
    street : str
    city : str
    state : str
    zip : int
    lat : float
    long : float
    city_pop : int
    job : str
    dob : str
    trans_num : str
    unix_time : int
    merch_lat : float
    merch_long : float

class ProcessDataRequestModel(BaseModel):
    transacciones: list[Transacciones]

    