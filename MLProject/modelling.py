import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import os
import joblib

def train():
    # 1. Setup Argparse untuk parameter MLProject
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # 2. Inisialisasi DagsHub (Ganti dengan kredensial Anda)
    # Gunakan nama repo yang sesuai dengan Kriteria 1/2
    repo_owner = "mrestuilahi1405"
    repo_name = "Eksperimen_SML_Muhammad-Restu-Ilahi"
    
# Cek apakah sedang berjalan di GitHub Actions
    if os.getenv("GITHUB_ACTIONS") == "true":
        # Jika di CI, jangan pakai dagshub.init interaktif
        # Langsung tembak URI Tracking-nya
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    else:
        # Jika di lokal, biarkan interaktif seperti biasa
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.set_experiment("CI_Workflow_ReTraining")

    # 3. Load Data Preprocessing
    # Path disesuaikan: di Kriteria 3, dataset ada di dalam folder MLProject
    data_path = "test_data.csv" # Sesuai struktur folder yang kita bahas
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. MLflow Logging
    with mlflow.start_run(run_name="MLProject_CI_Run"):
        # Log parameter dari argparse
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Training
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        # Log Metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # Log Model ke MLflow
        mlflow.sklearn.log_model(model, "model")

        # Log Data ke MLflow
        mlflow.log_artifact(data_path, "dataset")
        
        # Simpan lokal untuk keperluan build-docker nanti
        joblib.dump(model, "model.pkl")

        print(f"Training Selesai (MLProject)!")
        print(f"Params: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        print(f"Metrics: F1={f1:.4f}, Accuracy={acc:.4f}")

if __name__ == "__main__":
    train()