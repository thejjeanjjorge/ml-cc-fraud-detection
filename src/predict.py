"""Batch inference for the credit card fraud model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "robust_scaler.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "scored_transactions.csv"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALED_COLUMNS = ["Time", "Amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score credit card transactions with the trained fraud model.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a CSV containing transaction features.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=f"Path for scored CSV output. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help=f"Path to trained model artifact. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--scaler",
        default=DEFAULT_SCALER_PATH,
        type=Path,
        help=f"Path to fitted scaler artifact. Default: {DEFAULT_SCALER_PATH}",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        type=Path,
        help=f"Path to model metadata artifact. Default: {DEFAULT_METADATA_PATH}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fraud probability threshold. Uses model metadata when omitted.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    if path.exists():
        return joblib.load(path)
    return {}


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    return df.loc[:, FEATURE_COLUMNS].copy()


def score_dataframe(
    raw_df: pd.DataFrame,
    model,
    scaler,
    threshold: float,
) -> pd.DataFrame:
    features = validate_input(raw_df)
    features.loc[:, SCALED_COLUMNS] = scaler.transform(features[SCALED_COLUMNS])

    fraud_probability = model.predict_proba(features)[:, 1]
    fraud_prediction = (fraud_probability >= threshold).astype(int)

    scored_df = raw_df.copy()
    scored_df["fraud_probability"] = fraud_probability
    scored_df["fraud_prediction"] = fraud_prediction
    scored_df["threshold"] = threshold
    return scored_df


def score_transactions(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    threshold: float | None,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = load_metadata(metadata_path)

    decision_threshold = threshold
    if decision_threshold is None:
        decision_threshold = float(metadata.get("threshold", 0.5))

    raw_df = pd.read_csv(input_path)
    scored_df = score_dataframe(raw_df, model, scaler, decision_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    return scored_df


def main() -> None:
    args = parse_args()
    scored_df = score_transactions(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        scaler_path=args.scaler,
        metadata_path=args.metadata,
        threshold=args.threshold,
    )

    fraud_count = int(scored_df["fraud_prediction"].sum())
    print(f"Scored {len(scored_df):,} transactions")
    print(f"Predicted fraud: {fraud_count:,}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
