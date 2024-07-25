import config
import pickle
import gzip 
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler


def transform_dataframe(data)-> pd.DataFrame:
    data = pd.DataFrame(data)

    return data

def load_data():
    data = pd.read_csv('data/fraudTrain.csv')
    return data


def split_data(data):
    print(data.shape)

    X = data.drop(columns=['is_fraud'])
    Y = data['is_fraud']
    x_train , x_test , y_train , y_test = train_test_split(X , Y , shuffle=True , test_size=0.2 , random_state=42)

    return x_train , x_test , y_train , y_test

def scaler_data(x_train , x_test):
    scaler = StandardScaler()
    x_train_proced = scaler.fit_transform(x_train)
    x_test_proced = scaler.transform(x_test)

    return x_train_proced , x_test_proced


def search_best_param(model):
    best_model = GridSearchCV(
    estimator=model ,
    param_grid=config.PARAM_GRID ,
    n_jobs= -1 ,
    scoring='accuracy' ,
    cv = 5,
    verbose=1
    )

    return best_model

def load_model(model):

    joblib.dump(model ,"Model/model.pkl")