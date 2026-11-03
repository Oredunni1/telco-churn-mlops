# import libraries
import pandas as pd
from pathlib import Path

# Define paths correctly
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_PROCESS_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESS_FILE = DATA_PROCESS_DIR / "telco_clean.parquet"

def load_data():
    """Load and clean the Telco churn dataset into a validated parquet."""
    # read dataset
    df = pd.read_csv(DATA_RAW)

    # coerces TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Blank TotalCharges corresponds to tenure=0 (new customers not yet billed),
    # so fill with 0 rather than impute a fabricated charge.
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # drop the customerID
    df = df.drop(columns='customerID')

    # Map `Churn` to 0/1
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # return df
    return df

def build_processed_dataset():
    # Ensure the processed directory exists before saving
    DATA_PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    # --- DATA QUALITY & INTEGRITY CHECKS ---
    
    # 1. Row Count Check (Telco dataset raw has 7043 rows)
    expected_rows = 7043
    if len(df) != expected_rows:
        raise ValueError(f"Row count mismatch: expected {expected_rows}, got {len(df)}")
    
    # 2. Data Type Check (Ensure TotalCharges successfully coerced to float64)
    if df["TotalCharges"].dtype != "float64":
        raise ValueError(f"TotalCharges dtype is {df['TotalCharges'].dtype}, expected float64")
    
   # 3. Churn Rate Check (Guards against mapping or data drift bugs, expecting ~26.5%)
    churn_rate = df["Churn"].mean()
    min_bound, max_bound = 0.25, 0.28
    if not (min_bound <= churn_rate <= max_bound):
        raise ValueError(
            f"Churn rate validation failed! Calculated rate {churn_rate:.4f} "
            f"falls outside the expected bounds [{min_bound}, {max_bound}]"
        )
    # Save to parquet only if all validation checks pass
    df.to_parquet(DATA_PROCESS_FILE)
    
    print(f"✓ Data validation passed successfully:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - TotalCharges dtype: {df['TotalCharges'].dtype}")
    print(f"  - Churn rate: {churn_rate * 100:.2f}% (within expected bounds)")
    print(f"Processed data safely written to {DATA_PROCESS_FILE}")


if __name__ == "__main__":
    build_processed_dataset()