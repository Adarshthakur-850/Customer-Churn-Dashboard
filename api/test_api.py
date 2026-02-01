import requests
import json
customer_sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85"
}
def test_api():
    url = "http://127.0.0.1:8000/predict"
    try:
        print("Sending request to API...")
        response = requests.post(url, json=customer_sample)
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API is running (run 'python api/main.py' in a separate terminal)")
if __name__ == "__main__":
    test_api()