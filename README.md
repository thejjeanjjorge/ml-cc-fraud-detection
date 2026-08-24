# ml-cc-fraud-detection
ML Credit Card Fraud Detection System

## Data Access

The credit card fraud dataset is not committed to this repository because it is too large for GitHub. The project downloads it at runtime using KaggleHub and reads it from KaggleHub's local cache.

```python
from data.load_data import load_credit_card_data

df = load_credit_card_data()
```

If KaggleHub prompts for authentication, configure your local Kaggle API credentials first. The raw dataset file should stay out of Git.

## Project Structure

```text
.
|-- configs/                 # Experiment and pipeline configuration files
|-- data/
|   |-- external/            # Third-party reference data
|   |-- interim/             # Intermediate transformed data
|   |-- processed/           # Final model-ready datasets
|   `-- raw/                 # Original immutable data
|-- docs/                    # Project documentation
|-- models/                  # Trained model artifacts
|-- notebooks/               # Clean/reproducible notebooks
|-- reports/
|   `-- figures/             # Generated charts and visual outputs
`-- tests/                   # Unit and integration tests
```

The exploratory notebook currently lives at `notebooks/eda.ipynb`.
