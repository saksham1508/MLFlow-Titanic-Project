# Model training script for Titanic dataset
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_titanic

def train_model(data_csv, model_path='models/titanic_model.pkl', n_estimators=100, max_depth=5):
    """
    Train a RandomForest model on Titanic data, save model, and log with MLflow.
    Args:
        data_csv (str): Path to Titanic CSV file.
        model_path (str): Path to save trained model.
        n_estimators (int): Number of trees in RandomForest.
        max_depth (int): Max depth of trees.
    Returns:
        float: Validation accuracy.
    """
    df = pd.read_csv(data_csv)
    df = preprocess_titanic(df)
    X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1, errors='ignore')
    y = df['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_metric('val_accuracy', acc)
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, model_path)
        print(f"Validation Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Titanic Survival Model")
    parser.add_argument('--data', required=True, help='Path to Titanic CSV file')
    parser.add_argument('--model', default='models/titanic_model.pkl', help='Path to save trained model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RandomForest')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth of trees')
    args = parser.parse_args()
    train_model(args.data, args.model, args.n_estimators, args.max_depth)
