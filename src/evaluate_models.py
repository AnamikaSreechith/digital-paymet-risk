
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    classification_report
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
RESULT_PATH = BASE_DIR / "reports"

FIGURES_PATH = RESULT_PATH / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


# Selected XGBoost threshold
THRESHOLD = 0.97


def evaluate_model():

    # Load test dataset
    test_df = pd.read_csv(
        PROCESSED_PATH / "test_features.csv"
    )

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    # Load final trained XGBoost model
    model = joblib.load(
        MODEL_PATH / "final_xgboost_model.pkl"
    )

    # Generate fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply selected threshold
    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Calculate evaluation metrics
    metrics = {
        "Model": "XGBoost",
        "Threshold": THRESHOLD,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-Score": f1_score(y_test, predictions),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        )
    }

    # Display metrics
    print("\nFINAL XGBOOST EVALUATION")
    print("-" * 40)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Genuine", "Fraud"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Genuine", "Fraud"]
    )

    disp.plot(cmap="Blues", values_format="d")

    plt.title("XGBoost - Final Confusion Matrix")
    plt.savefig(
        FIGURES_PATH / "final_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"XGBoost (AP={metrics['PR-AUC']:.4f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        FIGURES_PATH / "xgboost_pr_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Save evaluation results
    results_df = pd.DataFrame([metrics])

    results_df.to_csv(
        RESULT_PATH / "final_evaluation.csv",
        index=False
    )

    print("\nEvaluation results saved successfully!")


if __name__ == "__main__":
    evaluate_model()
