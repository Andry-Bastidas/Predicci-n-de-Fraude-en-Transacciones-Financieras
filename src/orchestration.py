import utils
from processing_data import transform_data
from sqlalchemy.sql import insert
from db import engine , ReportModel

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sqlmodel import Session
from prefect import flow , task
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_recall_fscore_support, 
                             precision_score, recall_score, roc_auc_score , log_loss)
from sklearn.model_selection import GridSearchCV , train_test_split 
from sklearn.ensemble import GradientBoostingClassifier


@task(
      retries= 3 , retry_delay_seconds=5,
      name="Load data",
      tags=["Load data"]  
)
def load_data():
    data = utils.load_data()

    return data
        

@task(retries= 3 , retry_delay_seconds=5,
      name="Processing data",
      tags=["Transform data"]  
      )
def processing_data(data):
    data = transform_data(data)

    return data

@task(
    retries=3,
    retry_delay_seconds=5,
    name="Split data",
    tags=["split_data"]
)

def split_data(data):
    x_train , x_test , y_train , y_test = utils.split_data(data)
    return x_train , x_test , y_train , y_test

@task(
    retries=3,
    retry_delay_seconds=5,
    name="Train Model",
    tags=["train_model"]
)

def train_model(
      x_train:np.array,
      x_test:np.array,
      y_train:np.array,
      y_test:np.array

):

      session = Session(engine)
      model_x = GradientBoostingClassifier()
      grid_search = utils.search_best_param(model_x,x_train,y_train)

      model = grid_search.best_estimator_

      metrics = []
      
      y_train_pred_proba = model.predict_proba(x_train)
      y_test_pred_proba = model.predict_proba(x_test)

      y_train_proba_pos = y_train_pred_proba[:, 1]
      y_test_proba_pos = y_test_pred_proba[:, 1]

      roc_auc_score_train = round(
                  roc_auc_score(
                  y_train , y_train_proba_pos , average='weighted'
                  ),
                  2
            )

      roc_auc_score_test = round(
                  roc_auc_score(
                  y_test , y_test_proba_pos , average= 'weighted'
                  )
                  ,2
            )
      
      print("ROC AUC Score Train : " , roc_auc_score_train)
      print("ROC AUC Score Test : " , roc_auc_score_test)

      metrics.extend([roc_auc_score_train , roc_auc_score_test])

      y_train_pred = model.predict(x_train)
      y_test_pred = model.predict(x_test)

      score_train = accuracy_score(y_train , y_train_pred)
      score_test = accuracy_score(y_test , y_test_pred)

      log_loss_train = log_loss(y_train , y_train_pred)
      log_loss_test =  log_loss(y_test , y_test_pred)

      (
            precision_train,
            recall_train,
            fscore_train,
            support_train

      ) = precision_recall_fscore_support(
           
            y_train,
            y_train_pred,
            average="weighted"
      )

      (
            precision_test,
            recall_test,
            fscore_test,
            support_test

      ) = precision_recall_fscore_support(
           
            y_test,
            y_test_pred,
            average="weighted"
      )

      try:
           
           print(f"Accuracy Score Train : {score_train}")
           print(f"Accuracy Score Test : {score_test}")
           print(f"Params of model : {grid_search.best_params_}")
           assert  score_train > 0.86
           assert  score_test > score_train
           
           print("El modelo tuvo un rendimiento excelente")

      except AssertionError as e:
           print(f"Error : {e}")
           
           print("El modelo tuvo un mal rendimiento")

      metrics.extend(
                  [
                  round(score_train,2),
                  round(score_test, 2),
                  round(precision_train,2),
                  round(precision_test,2),
                  round(recall_train,2),
                  round(recall_test,2),
                  round(fscore_train,2),
                  round(fscore_test,2),
                  round(log_loss_train,2),
                  round(log_loss_test,2)
                  ]
            )
      
      reportmodel = ReportModel(
           
            date_time = datetime.now(),
            score_train = float(score_train) ,
            score_test = float(score_test),
            precision_train = float(precision_train),
            precision_test = float(precision_test),
            recall_train = float(recall_train),
            recall_test = float(recall_test),
            fscore_train = float(fscore_train),
            fscore_test = float(fscore_test),
            log_loss_train = float(log_loss_train),
            log_loss_test = float(log_loss_test)
            
        )
      
      
      
      model_report_train = classification_report(y_train , y_train_pred)
      model_report_test = classification_report(y_test , y_test_pred)

      print("Classification Report for Train:\n", model_report_train)
      print("Classification Report for Train:\n", model_report_test)

      session.add(reportmodel)

      session.commit()
      session.close()

      return model

@task(
      retries=3,
      retry_delay_seconds=5,
      name="Export_model",
      tags=["export model"]    
)
def export_model(model):
     utils.load_model(model)

@flow
def main_flow():
      data = load_data()
      data_pro = processing_data(data)
      x_train , x_test , y_train , y_test = split_data(data_pro)

      model = train_model(
           x_train,
           x_test,
           y_train,
           y_test
      )

      export_model(model)

main_flow()
      



