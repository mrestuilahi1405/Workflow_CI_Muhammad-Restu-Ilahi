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
import shutil

def train():
    # 1. Setup Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=104)
    parser.add_argument("--max_depth", type=int, default=27)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    args = parser.parse_args()

    # 2. Inisialisasi Tracking
    repo_owner = "mrestuilahi1405"
    repo_name = "Eksperimen_SML_Muhammad-Restu-Ilahi"
    
    if os.getenv("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    else:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    # 3. Load Data
    data_path = "credit_risk_clean.csv" 
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. MLflow Logging & Registration
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth, 
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # PENTING: Daftarkan model ke Registry agar bisa ditarik build-docker
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model_credit_risk",
            registered_model_name="model_credit_risk"
        )

        base_artifacts_path = "../artifacts"
        model_save_path = os.path.join(base_artifacts_path, "model_credit_risk")

        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path) # Hapus jika sudah ada
            
        os.makedirs(base_artifacts_path)

        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_save_path
        )

        data_save_path = os.path.join(data_path, "dataset")
        
        shutil.copy(data_path, os.path.join(base_artifacts_path, data_save_path))

        mlflow.log_artifact(data_path, "dataset")

        print(f"✅ Paket Lengkap (Model & Data) siap di: {base_artifacts_path}")
        print(f"✅ Training & Registration Selesai! F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()