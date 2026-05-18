import dagshub
import mlflow
import mlflow.sklearn

import json
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

dagshub.init(
    repo_owner='Adiwid20',
    repo_name='loan-approval-clasification',
    mlflow=True
)

mlflow.set_experiment("Loan_Approval_Tuning")


def run_modeling_tuning():

    print("MEMUAT DATASET")

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    df_pre = pd.read_csv(
        "dataset/loan_data_preprocessed.csv"
    )

    X = df_pre.drop("loan_status", axis=1)
    y = df_pre["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Melakukan balancing data dengan SMOTE...")

    smote = SMOTE(random_state=42)

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    n_estimators_range = np.linspace(
        10,
        200,
        5,
        dtype=int
    )

    max_depth_range = np.linspace(
        1,
        20,
        5,
        dtype=int
    )

    total_experiment = (
        len(n_estimators_range)
        * len(max_depth_range)
    )

    print(f"Total Experiment : {total_experiment}")

    best_accuracy = 0
    best_model = None
    best_params = {}
    best_metrics = {}

    experiment_id = 1

    with mlflow.start_run(
        run_name="RandomForest_Manual_Tuning"
    ):

        for n_estimators in n_estimators_range:

            for max_depth in max_depth_range:

                print(f"\nEXPERIMENT {experiment_id}")

                with mlflow.start_run(
                    run_name=f"RF_Run_{experiment_id}",
                    nested=True
                ):

                    model = RandomForestClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        random_state=42,
                        n_jobs=-1
                    )

                    model.fit(
                        X_train_balanced,
                        y_train_balanced
                    )

                    y_pred = model.predict(X_test)

                    y_prob = model.predict_proba(X_test)[:, 1]

                    accuracy = accuracy_score(
                        y_test,
                        y_pred
                    )

                    precision = precision_score(
                        y_test,
                        y_pred
                    )

                    recall = recall_score(
                        y_test,
                        y_pred
                    )

                    f1 = f1_score(
                        y_test,
                        y_pred
                    )

                    roc_auc = roc_auc_score(
                        y_test,
                        y_prob
                    )

                    print(f"Accuracy : {accuracy:.4f}")

                    params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth)
                    }

                    metrics = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "roc_auc": roc_auc
                    }

                    mlflow.log_params(params)

                    mlflow.log_metrics(metrics)


                    cm = confusion_matrix(
                        y_test,
                        y_pred
                    )

                    plt.figure(figsize=(6, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt='d',
                        cmap='Blues'
                    )

                    plt.title(
                        f"Confusion Matrix {experiment_id}"
                    )

                    cm_path = (
                        artifact_dir /
                        f"cm_{experiment_id}.png"
                    )

                    plt.savefig(cm_path)

                    plt.close()

                    mlflow.log_artifact(
                        str(cm_path),
                        artifact_path="plots"
                    )

                    report = classification_report(
                        y_test,
                        y_pred,
                        output_dict=True
                    )

                    json_path = (
                        artifact_dir /
                        f"metrics_{experiment_id}.json"
                    )

                    with open(json_path, "w") as f:

                        json.dump(
                            report,
                            f,
                            indent=4
                        )

                    mlflow.log_artifact(
                        str(json_path),
                        artifact_path="metrics"
                    )


                    input_example = (
                        X_train_balanced.iloc[:5]
                    )

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name="LoanApprovalModel",
                        input_example=input_example
                    )


                    if accuracy > best_accuracy:

                        best_accuracy = accuracy

                        best_model = model

                        best_params = params

                        best_metrics = metrics

                experiment_id += 1

        print("\nBEST MODEL")

        print(best_params)

        print(best_metrics)

        mlflow.log_params(best_params)

        mlflow.log_metrics(best_metrics)

        print(
            "Semua experiment berhasil "
            "tersimpan ke DagsHub"
        )


if __name__ == "__main__":

    run_modeling_tuning()