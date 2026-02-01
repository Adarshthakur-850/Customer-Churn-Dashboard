import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
def load_data():
    logging.info("Loading processed data...")
    X_train = pd.read_csv(PROCESSED_PATH / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_PATH / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_PATH / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED_PATH / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    }
    logging.info(f"--- {model_name} Performance ---")
    logging.info(f"Accuracy: {metrics['Accuracy']:.4f}")
    logging.info(f"ROC AUC: {metrics['ROC AUC']:.4f}")
    return metrics
def train_models():
    X_train, X_test, y_train, y_test = load_data()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    best_model = None
    best_score = 0
    best_name = ""
    all_metrics = []
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)
        if metrics["ROC AUC"] > best_score:
            best_score = metrics["ROC AUC"]
            best_model = model
            best_name = name
    logging.info(f"Best Model: {best_name} with ROC AUC: {best_score:.4f}")
    joblib.dump(best_model, MODELS_PATH / "best_model.pkl")
    logging.info(f"Saved best model to {MODELS_PATH / 'best_model.pkl'}")
    with open(MODELS_PATH / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    logging.info(f"Saved metrics to {MODELS_PATH / 'metrics.json'}")
if __name__ == "__main__":
    train_models()