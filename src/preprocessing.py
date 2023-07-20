import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyyamlconfig import load_config

path = os.path.dirname(os.getcwd())

# Load the YAML file
params = load_config(path + '/params.yaml')

data = pd.read_csv(path + "/data/raw/Customer_Churn.csv")

del(data['Time'])

X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

import sklearn.preprocessing as prep

scaler = params['scaling']['scaler']
X_scaled = scaler.fit_transform(X)

