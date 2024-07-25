import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OrdinalEncoder
import sys

def transform_data(data:pd.DataFrame):
    # Equilibracion de clases

    rus = RandomUnderSampler(random_state=42)
    X = data.drop("is_fraud",axis=1)
    Y = data.is_fraud 

    X_over, y_over = rus.fit_resample(X,Y)
    y_over.unique()

    data = pd.concat([X_over, y_over] , axis = 1)

    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
    data['date'] = data['trans_date_trans_time'].dt.date
    data['time'] = data['trans_date_trans_time'].dt.time

    data['date'] = pd.to_datetime(data['date'] , format='%Y-%m-%d')
    data['time'] = pd.to_datetime(data['time'] , format ='%H:%M:%S')

    data['year'] = data['date'].dt.year
    data['week'] = data['date'].dt.weekday
    data['month'] = data['date'].dt.month
    data['hour'] = data['time'].dt.hour

    data["hour_sin"] = np.sin( 2 * np.pi * (data["hour"] / 24))
    data["hour_cos"] = np.cos( 2 * np.pi * (data["hour"] / 24))

    data["week_sin"] = np.sin( 2 * np.pi * (data["week"] / 7))
    data["week_cos"] = np.cos( 2 * np.pi * (data["week"] / 7))

    data['month_sin'] = np.sin(2 * np.pi * (data['month'] / 12.0))
    data['month_cos'] = np.cos(2 * np.pi * (data['month'] / 12.0))


    data.drop(columns=['date' , 'time'] , inplace=True)

    card_transaction_counts = data['cc_num'].value_counts()
    data['card_transaction_count'] = data['cc_num'].map(card_transaction_counts)
    data.drop(columns=['cc_num'] , axis = 1 , inplace=True)

    data.drop(columns=["Unnamed: 0", 'first','last','street','trans_num','city','state','merchant','job','trans_date_trans_time'], inplace=True)

    data['dob'] = pd.to_datetime(data['dob'] , format='%Y-%m-%d')
    data['year2'] = data['dob'].dt.year
    data['age'] = data[['year2' , 'year']].apply(
        lambda x : x['year'] - x['year2'] , axis = 1
    )

    data.drop(columns=['year2' ,'dob' , 'year' , 'month' , 'hour' , 'week'] , axis= 1, inplace=True)

    columnas = ['hour_sin' , 'hour_cos' , 'week_sin', 'week_cos' , 'month_sin', 'month_cos' , 'age'] + [ col for col in data.columns if col not in ['hour_sin' , 'hour_cos' , 'week_sin', 'week_cos' , 'month_sin', 'month_cos' , 'age' ]]
    data = data[columnas]

    encoder = OrdinalEncoder()
    data_categoric = data.select_dtypes(exclude = np.number)
    data_become_numeric = encoder.fit_transform(data_categoric)
    data[['category' , 'gender']] = data_become_numeric
    
    return data