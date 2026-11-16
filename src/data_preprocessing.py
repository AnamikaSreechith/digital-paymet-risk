
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def preprocess_data():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Remove duplicate transactions
    df = df.drop_duplicates()

    print("After removing duplicates:", df.shape)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale only Time and Amount
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    columns_to_scale = ["Time", "Amount"]

    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    # Save scaler
    joblib.dump(
        scaler,
        MODEL_PATH / "standard_scaler.pkl"
    )

    # Save processed datasets
    X_train.assign(Class=y_train).to_csv(
        PROCESSED_PATH / "train_processed.csv",
        index=False
    )

    X_test.assign(Class=y_test).to_csv(
        PROCESSED_PATH / "test_processed.csv",
        index=False
    )

    print("\nPreprocessing completed!")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    print("\nTraining fraud distribution:")
    print(y_train.value_counts())

    print("\nTesting fraud distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    preprocess_data()
