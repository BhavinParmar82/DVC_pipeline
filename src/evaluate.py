import os
import joblib
import click
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
#@click.argument('output_file', type=click.Path())

def evaluate(input_file):
    path = os.path.join(os.getcwd())
    model_file_path = os.path.join(path, input_file)
    #output_file_path = os.path.join(path, output_file)

    test_data = pd.read_csv(path + '/data/processed/test_data.csv')
    
    model = joblib.load(model_file_path)
    y_predict = model.predict(test_data.drop(columns='target'))
    
    # Compute evaluation metrics
    cm = confusion_matrix(test_data['target'], y_predict)
    f1 = f1_score(test_data['target'], y_predict)

    print("Confusion Matrix:")
    print(cm)
    print("F1 Score:", f1)

    #cm.to_csv(output_file_path)

if __name__ == "__main__":
    evaluate()
    
        
    