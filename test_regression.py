import sys
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from regression_preprocessing import preprocess_data

def run_regression_test(test_csv_path):
    print("=" * 60)
    print("Running Unseen REGRESSION Evaluation")
    print("=" * 60)
    
    # Load raw data
    df = pd.read_csv(test_csv_path)
    
    # Apply saved preprocessing (is_training=False prevents data leakage)
    X_processed, y_true = preprocess_data(df, is_training=False)
    
    # Drop demographic noise identically to training
    features_to_drop = ['region', 'gender', 'disability', 'imd_band', 'age_band'] 
    X_selected = X_processed.drop(columns=features_to_drop, errors='ignore')
    
    # Load and apply the saved scaler
    with open(os.path.join("saved_models", "reg_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_selected)
    
    # Load models
    with open(os.path.join("saved_models", "reg_model_lr.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join("saved_models", "reg_model_rf.pkl"), "rb") as f:
        rf_model = pickle.load(f)
        
    models = {"Linear Regression": lr_model, "Random Forest": rf_model}
    
    for name, model in models.items():
        preds = model.predict(X_scaled)
        print(f"\n--- {name} ---")
        if y_true is not None:
            mse = mean_squared_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            print(f"MSE: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")
        else:
            print("No 'score' column found in test data. Outputting raw predictions only.")
            print(preds[:10])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Provide the path to the unseen regression CSV.")
        sys.exit(1)
    run_regression_test(sys.argv[1])