from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import re
import ast

def get_model(model_type, input_dim=None):
    if model_type == 'LR':
        model = LinearRegression()
    elif model_type == 'SVR':
        model = SVR(kernel='rbf')
    elif model_type == 'RFR':
        model = RandomForestRegressor()
    elif model_type == 'GBR':
        model = GradientBoostingRegressor()
    elif model_type.startswith('MLPR'):
        pattern = r'\((.*?)\)'
        tuple_match = re.search(pattern, model_type)
        if tuple_match:
            tuple_str = tuple_match.group(1)
            hidden_layer_sizes = ast.literal_eval(tuple_str)
        else:
            raise ValueError("Invalid MLPR architecture.")
        
        model = build_NN(hidden_layer_sizes=hidden_layer_sizes, dropout_rate=0.0, input_dim=input_dim)
        #model = MLPRegressor(hidden_layer_sizes, random_state=1, max_iter=1000, early_stopping=True, validation_fraction=0.2)
    elif model_type.startswith('LASSO'):
        pattern = r"LASSO\((\d+(\.\d+)?)\)"
        match = re.search(pattern, model_type)
        if match:
            alpha = float(match.group(1))
        else:
            raise ValueError("Invalid Lasso alpha parameter.")
        model = Lasso(alpha=alpha, max_iter=2000)
    else:
        raise ValueError("Invalid model_type")
        
    return model   

# Define the model building function
def build_NN(input_dim, hidden_layer_sizes, dropout_rate):
    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for layer_size in hidden_layer_sizes[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model