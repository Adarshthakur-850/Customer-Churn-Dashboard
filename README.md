# Customer Churn Dashboard

A machine learning–powered dashboard that predicts whether a customer is likely to churn using historical customer data. The project combines data analysis, model training, an API layer, and visual insights to help businesses identify at-risk customers and take preventive action.

---

## Problem Statement

Customer churn is a major challenge for subscription-based and service businesses. Losing customers directly impacts revenue. This project builds a predictive system that analyzes customer behavior and forecasts churn probability, enabling proactive retention strategies.

---

## Objectives

* Analyze customer data to identify churn patterns
* Train a machine learning model to predict churn
* Save and reuse the trained model for predictions
* Provide an API interface for real-time predictions
* Visualize insights through a dashboard

---

## Tech Stack

* **Python**
* **Pandas, NumPy** – Data processing
* **Scikit-learn** – Machine learning models
* **Matplotlib / Seaborn** – Visualization
* **FastAPI** – API layer
* **Jupyter Notebook** – EDA and experimentation

---

## Project Structure

```
Customer-Churn-Dashboard/
│
├── api/            # API to serve churn predictions
├── data/           # Raw and processed datasets
├── models/         # Saved trained ML models (.pkl)
├── notebooks/      # EDA, preprocessing, model training
├── src/            # Core scripts for training and preprocessing
├── requirements.txt
└── README.md
```

---

## Workflow

```
Dataset → Data Cleaning → Feature Engineering → Model Training
        → Model Saving → API Integration → Dashboard Visualization
```

---

## Model Performance

> (Update these values from your notebook results)

* Accuracy: XX%
* ROC-AUC Score: XX
* Precision: XX
* Recall: XX
* Confusion Matrix: Available in notebooks

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Start the API server

```bash
uvicorn api.main:app --reload
```

### 4. Open notebooks

Use Jupyter Notebook to explore EDA and training steps:

```bash
jupyter notebook
```

---

## Features

* End-to-end churn prediction pipeline
* Structured ML workflow
* Model persistence for reuse
* API for real-time predictions
* Visual data analysis and insights

---

## Screenshots

(Add dashboard and output screenshots in an `assets/` folder and place them here)

---

## Future Improvements

* Dockerization for deployment
* CI/CD pipeline integration
* Live dashboard hosting
* Hyperparameter tuning and model comparison

---

## Use Case

This system can be used by businesses to:

* Identify customers likely to churn
* Take targeted retention actions
* Improve customer lifetime value
* Reduce revenue loss

---

## Author

**Adarsh Thakur**
Machine Learning | Data Science | DevOps Enthusiast
