from data_loader import load_and_merge_data
from preprocessing import preprocess_data
from models import train_and_evaluate, plot_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("1. Loading and merging datasets from the subfolder...")
    # The default data_dir in load_and_merge_data now points to your specific folder
    raw_df = load_and_merge_data()

    print("2. Preprocessing all features...")
    X_processed, y = preprocess_data(raw_df)

    print("3. Feature Selection...")
    # Dropping demographic noise. 
    # final_result is excluded here because it was already scrubbed from the source CSV.
    features_to_drop = [
        'region', 
        'gender', 
        'disability', 
        'imd_band', 
        'age_band'
    ] 
    
    # errors='ignore' prevents the script from crashing if a column is already missing
    X_selected = X_processed.drop(columns=features_to_drop, errors='ignore')

    print("4. Splitting and Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("5. Training Models & Evaluating...")
    predictions = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)

    print("6. Generating Plots...")
    plot_results(y_test, predictions)

if __name__ == "__main__":
    main()