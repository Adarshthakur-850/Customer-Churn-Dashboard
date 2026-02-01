import os
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
DATA_FILE = RAW_DATA_PATH / "Telco-Customer-Churn.csv"

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_data():
    """Downloads the Telco Customer Churn dataset."""
    
    if not RAW_DATA_PATH.exists():
        RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    if DATA_FILE.exists():
        logging.info(f"Dataset already exists at {DATA_FILE}")
        return

    logging.info(f"Downloading dataset from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        with open(DATA_FILE, 'wb') as f:
            f.write(response.content)
            
        logging.info(f"Dataset successfully saved to {DATA_FILE}")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_data()
