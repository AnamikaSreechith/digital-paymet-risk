
import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def train_models():

    # Load prepared datasets
    train_df = pd.read_csv(
        PROCESSED_PATH / "train_features.csv"
    )

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    print("Training dataset:", X_train.shape)

    # 1. Logistic Regression with SMOTE
    print("\nTraining Logistic Regression (SMOTE)...")

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(
        X_train, y_train
    )

    logistic_smote = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    logistic_smote.fit(X_resampled, y_resampled)

    joblib.dump(
        logistic_smote,
        MODEL_PATH / "logistic_smote.pkl"
    )

    # 2. Logistic Regression with class weights
    print("\nTraining Logistic Regression (Weighted)...")

    logistic_weighted = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    logistic_weighted.fit(X_train, y_train)

    joblib.dump(
        logistic_weighted,
        MODEL_PATH / "logistic_weighted.pkl"
    )

    # 3. XGBoost
    print("\nTraining XGBoost...")

    scale_pos_weight = (
        (y_train == 0).sum() /
        (y_train == 1).sum()
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)

    joblib.dump(
        xgb_model,
        MODEL_PATH / "xgboost_model.pkl"
    )

    joblib.dump(
        xgb_model,
        MODEL_PATH / "final_xgboost_model.pkl"
    )

    # 4. Isolation Forest
    print("\nTraining Isolation Forest...")

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,
        random_state=42,
        n_jobs=-1
    )

    isolation_model.fit(X_train)

    joblib.dump(
        isolation_model,
        MODEL_PATH / "isolation_forest.pkl"
    )

    print("\nAll models trained and saved successfully!")

    print("\nSaved models:")
    print("1. logistic_smote.pkl")
    print("2. logistic_weighted.pkl")
    print("3. xgboost_model.pkl")
    print("4. final_xgboost_model.pkl")
    print("5. isolation_forest.pkl")


if __name__ == "__main__":
    train_models()
