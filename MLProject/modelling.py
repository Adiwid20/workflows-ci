# pyrefly: ignore [missing-import]
import mlflow
import mlflow.sklearn
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings 


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", None)
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking ke: {MLFLOW_TRACKING_URI}")
else:
    print("MLflow tracking ke: mlruns/ lokal")

mlflow.set_experiment("Loan_Approval")

def run_modeling():
    warnings.filterwarnings("ignore")

    print("Memuat dataset...")
    
    if os.path.exists("loan_data_preprocessed.csv"):
        dataset_path = "loan_data_preprocessed.csv"
    else:
        dataset_path = "MLProject/loan_data_preprocessed.csv"
    
    df_pre = pd.read_csv(dataset_path)

    X = df_pre.drop('loan_status', axis=1)
    y = df_pre['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    input_example = X_train[0:5]

    with mlflow.start_run():

        mlflow.autolog()
        print("MLflow Autolog diaktifkan.")

        n_estimators = 20 
        max_depth = 10
        random_state = 42

        print("Melakukan balancing data (SMOTE)...")
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print("Melatih model Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train_balanced, y_train_balanced)

        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)


        mlflow.log_metric("accuracy", accuracy)


        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        
        model_dir = "saved_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model disimpan ke: {model_path}")

        
        mlflow.log_artifact(model_path, artifact_path="saved_model")
        
        print(f"\n--- HASIL EVALUASI ---")
        print(f"Akurasi Model pada Data Uji: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Diterima (0)', 'Ditolak (1)']))

if __name__ == "__main__":
    run_modeling()