import os

from sqlmodel import Field , SQLModel, create_engine
from sqlalchemy import TIMESTAMP


class ReportModel(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    date_time : TIMESTAMP
    score_train : int
    score_test : int
    precision_train : int
    precision_test : int
    recall_train : int
    recall_test: int 
    fscore_train : int
    fscore_test : int

DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL , echo=True)

def create_db_and_tables():

    SQLModel.metadata.create_all(engine)