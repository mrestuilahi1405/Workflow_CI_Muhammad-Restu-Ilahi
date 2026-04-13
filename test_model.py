import pandas as pd
import joblib
import pytest
from sklearn.metrics import accuracy_score
import os

def test_model_performance():
    # 1. Load Model & Data
    # Path disesuaikan dengan struktur folder submission nanti
    model_path = "model.pkl" 
    data_path = "test_data.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        pytest.skip("Model atau data uji tidak ditemukan, melewati tes.")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X_test = df.drop('loan_status', axis=1)
    y_test = df['loan_status']
    
    # 2. Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 3. Threshold Verification (Syarat Advance)
    threshold = 0.75
    print(f"Model Accuracy: {acc}")
    assert acc > threshold, f"Akurasi model {acc} di bawah threshold {threshold}!"

def test_data_consistency():
    # Memastikan jumlah fitur input konsisten
    data_path = "test_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Contoh: fitur harus ada 20 (sesuaikan dengan hasil preprocessing Anda)
        assert df.shape[1] > 1, "Dataset tidak memiliki fitur prediktor."