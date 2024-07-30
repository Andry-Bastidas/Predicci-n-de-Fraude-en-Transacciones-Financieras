import os

from sqlmodel import Field, Session, SQLModel, create_engine, select
from datetime import datetime

class ReportModel(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    date_time : datetime
    score_train : float
    score_test : float
    precision_train : float
    precision_test : float
    recall_train : float
    recall_test: float 
    fscore_train : float
    fscore_test : float
    log_loss_train : float
    log_loss_test : float

DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL , echo=True)

