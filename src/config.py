DATA_PATH_PROCESSED = 'data'

FILE_NAME_DATA_INPUT = 'data_processed.csv'

labels = {
    'No_fraud' : 0,
    'Is_Fraud' : 1
}

DEVELOPER_NAME = "Cristian"
MODEL_NAME = "GradientBoostingClassifier"


PARAM_GRID = {
    'learning_rate': [0.1, 0.01],
    'n_estimators': [50, 75 , 100, 125],
    'max_depth': range(4 , 9)
}
