import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_PATH = PROJECT_ROOT / "models"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
def get_feature_names(preprocessor):
    try:
        transformers = preprocessor.transformers_
        feature_names = []
        for name, transformer, columns in transformers:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(columns))
                else:
                    feature_names.extend(columns)
        return feature_names
    except Exception as e:
        logging.error(f"Error extracting feature names: {e}")
        return []
def analyze_feature_importance():
    logging.info("Loading model and preprocessor...")
    if not (MODELS_PATH / "best_model.pkl").exists():
        raise FileNotFoundError("Best model not found. Run train_model.py first.")
    model = joblib.load(MODELS_PATH / "best_model.pkl")
    preprocessor = joblib.load(MODELS_PATH / "preprocessor.pkl")
    feature_names = get_feature_names(preprocessor)
    importances = None
    if hasattr(model, "coef_"):
        importances = model.coef_[0]
        logging.info("Extracted coefficients from Linear Model.")
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        logging.info("Extracted feature importances from Tree Model.")
    else:
        logging.warning("Model does not support feature importance extraction directly.")
        return
    if len(feature_names) != len(importances):
        logging.warning(f"Feature names count ({len(feature_names)}) does not match importance count ({len(importances)}).")
        try:
             X_train_df = pd.read_csv(PROCESSED_PATH / "X_train.csv")
             feature_names = X_train_df.columns.tolist()
             if len(feature_names) == len(importances):
                 logging.info("Successfully matched features using X_train columns.")
        except Exception as e:
            logging.error(f"Fallback failed: {e}")
            return
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importance_df['AbsImportance'] = feature_importance_df['Importance'].abs()
    feature_importance_df = feature_importance_df.sort_values(by='AbsImportance', ascending=False)
    feature_importance_df.to_csv(MODELS_PATH / "feature_importance.csv", index=False)
    logging.info(f"Saved feature importance to {MODELS_PATH / 'feature_importance.csv'}")
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title(f"Top 20 Features - {type(model).__name__}")
    plt.xlabel("Coefficient / Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(MODELS_PATH / "feature_importance.png")
    logging.info(f"Saved plot to {MODELS_PATH / 'feature_importance.png'}")
if __name__ == "__main__":
    analyze_feature_importance()