import pickle
import os
from regression_data_loader import load_and_merge_data
from regression_preprocessing import preprocess_data
from regression_models import train_and_evaluate, plot_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    os.makedirs("saved_models", exist_ok=True)
    
    print("1. Loading and merging datasets from the subfolder...")
    raw_df = load_and_merge_data()

    print("2. Preprocessing all features...")
    X_processed, y = preprocess_data(raw_df, is_training=True)

    print("3. Feature Selection...")
    features_to_drop = ['region', 'gender', 'disability', 'imd_band', 'age_band'] 
    X_selected = X_processed.drop(columns=features_to_drop, errors='ignore')

    print("4. Splitting and Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for testing
    with open(os.path.join("saved_models", "reg_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("5. Training Models & Evaluating...")
    predictions = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    print("6. Generating Plots...")
    plot_results(y_test, predictions)

if __name__ == "__main__":
    main()