import pandas as pd
import os
import sklearn.preprocessing as prep
from pyyamlconfig import load_config

path = os.path.dirname(os.getcwd())

import yaml
params_file = os.path.join(path, 'params.yaml')
# Read the params.yaml file
with open(params_file, 'r') as file:
    params = yaml.safe_load(file)

def preprocessing():
    data_file = os.path.join(path, "data/raw/Customer_Churn.csv")
    data = pd.read_csv(data_file)

    del(data['Time'])

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    # Get the scaler name from the params.yaml
    scaler_name = params['scaling']['scaler']

    # Get the scaler class from sklearn.preprocessing
    scaler_class = getattr(prep, scaler_name)

    # Create an instance of the scaler
    scaler = scaler_class()

    # Fit and transform the data
    X_scaled = scaler.fit_transform(X)

    # Save the scaled data
    scaled_data = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_data.to_csv(path + "/data/processed/X_scaled_data.csv", index=False)

if __name__ == "__main__":
    preprocessing()