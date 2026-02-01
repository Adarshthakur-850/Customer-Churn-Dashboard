import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
def preprocess_data():
    logging.info("Loading data...")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    logging.info("Data cleaned. Encoding target variable...")
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [c for c in X.columns if c not in numeric_features]
    logging.info(f"Numeric features: {len(numeric_features)}")
    logging.info(f"Categorical features: {len(categorical_features)}")
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    logging.info("Saving processed data and pipeline...")
    X_train_df.to_csv(PROCESSED_PATH / "X_train.csv", index=False)
    X_test_df.to_csv(PROCESSED_PATH / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_PATH / "y_test.csv", index=False)
    joblib.dump(preprocessor, MODELS_PATH / "preprocessor.pkl")
    logging.info(f"Preprocessing complete. Preprocessor saved to {MODELS_PATH / 'preprocessor.pkl'}")
if __name__ == "__main__":
    preprocess_data()