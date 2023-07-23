import click
import pandas as pd
import os
import sklearn.preprocessing as prep
from pyyamlconfig import load_config
import yaml

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())

def preprocessing(input_file, output_file):
    path = os.path.join(os.getcwd())

    input_file_path = os.path.join(path, input_file)
    output_file_path = os.path.join(path, output_file)
    params_file = os.path.join(path, "params.yaml")

    # Read the params.yaml file
    with open(params_file, 'r') as file:
        params = yaml.safe_load(file)["scaling"]

    data = pd.read_csv(input_file_path)
    del(data['Time'])

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    # Get the scaler name from the params.yaml
    scaler_name = params["scaler"]

    # Get the scaler class from sklearn.preprocessing
    scaler_class = getattr(prep, scaler_name)

    # Create an instance of the scaler
    scaler = scaler_class()

    # Fit and transform the data
    X_scaled = scaler.fit_transform(X)

    # Save the scaled data
    scaled_data = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_data['target'] = y
    scaled_data.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    preprocessing()
