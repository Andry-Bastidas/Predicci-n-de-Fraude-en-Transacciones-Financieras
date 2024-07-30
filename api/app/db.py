import os

from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime


class PredictionsTransaccion(SQLModel, table=True):
    """
    Table Model for Prediction Tickets.

    This class represents a table in the database to store
    predictions associated with client names.
    """
    id: int | None = Field(default=None, primary_key=True)# Field se utiliza para proporcionar validaciones adicionales y configuraciones para el campo.
    client_name: str # columnas tipo str
    client_last_name : str
    client_cc_num : str
    trans_time : str
    prediction: str # columna tipo str

class ReportModel(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    date_time : datetime
    score_train : int
    score_test : int
    precision_train : int
    precision_test : int
    recall_train : int
    recall_test: int 
    fscore_train : int
    fscore_test : int
    log_loss_train : float
    log_loss_test : float

DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL , echo=True)

def create_db_and_tables():

    SQLModel.metadata.create_all(engine)
