import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, is_training=True):
    target = 'score'
    drop_cols = ['id_assessment', 'id_student', 'id_site', 'id_assess']
    features_df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    y = features_df[target] if target in features_df.columns else None
    X_raw = features_df.drop(columns=[target], errors='ignore')

    num_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_raw.select_dtypes(include=['object', 'bool']).columns.tolist()

    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    prep_path = os.path.join(model_dir, "reg_preprocessor.pkl")
    cols_path = os.path.join(model_dir, "reg_columns.pkl")

    if is_training:
        num_transformer = SimpleImputer(strategy='median')
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        
        preprocessor = ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ])
        
        X_processed = preprocessor.fit_transform(X_raw)
        
        with open(prep_path, "wb") as f:
            pickle.dump(preprocessor, f)
        with open(cols_path, "wb") as f:
            pickle.dump({"num": num_cols, "cat": cat_cols}, f)
            
        return pd.DataFrame(X_processed, columns=num_cols + cat_cols), y
    else:
        # Load unseen data
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
        with open(cols_path, "rb") as f:
            cols = pickle.load(f)
            
        # Ensure all columns exist
        for col in cols['num'] + cols['cat']:
            if col not in X_raw.columns:
                X_raw[col] = pd.NA
                
        X_raw = X_raw[cols['num'] + cols['cat']]
        X_processed = preprocessor.transform(X_raw)
        
        return pd.DataFrame(X_processed, columns=cols['num'] + cols['cat']), y