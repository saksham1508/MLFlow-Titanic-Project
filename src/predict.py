# Prediction script for Titanic model
import pandas as pd
import joblib
from utils import preprocess_titanic

def predict(input_csv, model_path, output_csv=None):
    """
    Predict survival on Titanic data using a trained model.
    Args:
        input_csv (str): Path to input CSV file with Titanic data.
        model_path (str): Path to trained model file (joblib/pkl).
        output_csv (str, optional): If provided, save predictions to this file.
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    df = pd.read_csv(input_csv)
    X = preprocess_titanic(df.copy())
    model = joblib.load(model_path)
    preds = model.predict(X)
    df['Prediction'] = preds
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df[['PassengerId', 'Prediction']]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Titanic Survival Prediction")
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--model', required=True, help='Path to trained model file (joblib/pkl)')
    parser.add_argument('--output', required=False, help='Path to save predictions CSV')
    args = parser.parse_args()
    result = predict(args.input, args.model, args.output)
    print(result.head())
