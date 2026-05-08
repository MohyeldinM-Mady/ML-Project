import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    target = 'score'
    
    # Drop ALL potential ID columns to prevent data leakage
    drop_cols = ['id_assessment', 'id_student', 'id_site', 'id_assess']
    features_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    y = features_df[target]
    X_raw = features_df.drop(columns=[target])

    num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_raw.select_dtypes(include=['object', 'bool']).columns

    # 1. Imputation (Handling missing values)
    num_imputer = SimpleImputer(strategy='median')
    X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_raw[cat_cols] = cat_imputer.fit_transform(X_raw[cat_cols])

    # 2. Categorical Encoding (Converting text to numbers)
    le = LabelEncoder()
    for col in cat_cols:
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    return X_raw, y