# src/features.py
"""Feature preprocessing: build the ColumnTransformer for the churn pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


NUM_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']



def build_preprocessor() -> ColumnTransformer:
    """Build the Preprocessor Pipeline
    NUM_COLS: All Numerical Columns stored in the NUM_COLS
    CAT_COLS: All Categorical Columns stored in the CAT_COLS,
        SeniorCitizen is int64 but represents a binary category (0=not senior, 1=senior). 
        Treated as categorical (one-hot + drop='if_binary') rather than numeric — the values are labels, 
        not measurements, and dropping one column avoids perfect collinearity with the intercept.
    Return: A ColumnTransformer
    """

    # Instantiate Numerical Pipeline
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Instantiate Categorical Pipeline
    # sparse_output: Enforces to output as numpy array rather than sparse matrix, for easier use in dataframes
    # handle_unknown: ignore, ensure when new category comes in, it should be ignore (set to 0)
    # drop='if_binary': for binary categoricals (gender, SeniorCitizen, Partner, Dependents,
    # PhoneService, PaperlessBilling), drop one column to avoid perfect multicollinearity
    # in linear models. Non-binary categoricals (Contract, InternetService, PaymentMethod,
    # MultipleLines, OnlineX/StreamingX) are unaffected — they still emit all levels.
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    # Set up the columnTransform
    return ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS)
    ])

if __name__ == "__main__":
    p = build_preprocessor()
    print(p)
    print(len(p.transformers))
    print([name for name, _, _ in p.transformers])
