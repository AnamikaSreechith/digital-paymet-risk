
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
FIGURES_PATH = BASE_DIR / "reports" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def explain_model():

    # Load final XGBoost model
    model = joblib.load(
        MODEL_PATH / "final_xgboost_model.pkl"
    )

    # Load test dataset
    test_df = pd.read_csv(
        PROCESSED_PATH / "test_features.csv"
    )

    X_test = test_df.drop("Class", axis=1)

    # Select representative sample
    X_shap = X_test.sample(
        n=min(1000, len(X_test)),
        random_state=42
    )

    print("Calculating SHAP values...")

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_shap)

    print("SHAP values calculated!")

    # 1. SHAP Beeswarm Plot
    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False
    )

    plt.title("SHAP Feature Impact on Fraud Detection")

    plt.savefig(
        FIGURES_PATH / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # 2. Global Feature Importance
    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False
    )

    plt.title("Global SHAP Feature Importance")

    plt.savefig(
        FIGURES_PATH / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # 3. Individual Transaction Explanation
    probabilities = model.predict_proba(X_test)[:, 1]

    # Select highest-risk transaction
    fraud_idx = probabilities.argmax()

    individual_transaction = X_test.iloc[[fraud_idx]]

    individual_shap = explainer(individual_transaction)

    shap.plots.waterfall(
        individual_shap[0],
        max_display=15,
        show=False
    )

    plt.title("Individual Transaction SHAP Explanation")

    plt.savefig(
        FIGURES_PATH / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nSHAP explainability completed!")

    print("Explained transaction index:", fraud_idx)
    print("Fraud probability:", probabilities[fraud_idx])


if __name__ == "__main__":
    explain_model()
