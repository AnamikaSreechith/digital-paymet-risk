
import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed"

# Expected model features (30)
FEATURE_COLUMNS = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)


def prepare_features():

    # Load preprocessed datasets
    train_df = pd.read_csv(
        PROCESSED_PATH / "train_processed.csv"
    )

    test_df = pd.read_csv(
        PROCESSED_PATH / "test_processed.csv"
    )

    # Separate features and target
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["Class"].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["Class"].copy()

    # Validate feature dimensions
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS

    # Save prepared datasets
    X_train.assign(Class=y_train).to_csv(
        PROCESSED_PATH / "train_features.csv",
        index=False
    )

    X_test.assign(Class=y_test).to_csv(
        PROCESSED_PATH / "test_features.csv",
        index=False
    )

    print("Feature preparation completed!")

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("Total features:", len(FEATURE_COLUMNS))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_features()
