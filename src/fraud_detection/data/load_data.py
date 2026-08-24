from pathlib import Path

import kagglehub
import pandas as pd


KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
CSV_FILENAME = "creditcard.csv"


def get_credit_card_csv_path() -> Path:
    """Download the Kaggle dataset if needed and return the local CSV path."""
    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    csv_path = dataset_path / CSV_FILENAME

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected dataset file was not found: {csv_path}")

    return csv_path


def load_credit_card_data() -> pd.DataFrame:
    """Load the credit card fraud dataset from the KaggleHub cache."""
    return pd.read_csv(get_credit_card_csv_path())
