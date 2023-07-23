from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yaml
import os
import pandas as pd
import joblib
import click

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())

def train(input_file, output_file):
    path = os.path.join(os.getcwd())
    input_file_path = os.path.join(path, input_file)
    model_file_path = os.path.join(path, output_file)
    params_file = os.path.join(path, "params.yaml")

    # Read the params.yaml file
    with open(params_file, 'r') as file:
        params = yaml.safe_load(file)["train"]
        
    data = pd.read_csv(input_file_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['target']), data['target'], random_state=params['seed'], test_size=params['ratio'])
    
    # Save the train data
    train_data = pd.DataFrame(X_train, columns=data.drop(columns=['target']).columns)
    train_data['target'] = y_train
    train_data.to_csv("data/processed/train_data.csv", index=False)
    
    # Save the test data
    test_data = pd.DataFrame(X_test, columns=data.drop(columns=['target']).columns)
    test_data['target'] = y_test
    test_data.to_csv("data/processed/test_data.csv", index=False)
    
    rfc = RandomForestClassifier(n_estimators = params['estimator'])
    rfc.fit(X_train, y_train)
    
    # Save the trained model to the specified file path
    with open(model_file_path, 'wb') as file:
        joblib.dump(rfc, file)
        
if __name__ == "__main__":
    train()
    