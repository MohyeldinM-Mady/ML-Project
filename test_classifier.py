import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
#                   CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "saved_models")


def load_artifacts():
    """Load preprocessor, label encoder, feature columns."""
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
        feature_info = pickle.load(f)
    return preprocessor, label_encoder, feature_info


def load_models():
    """Load best models."""
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_best.pkl"):
            model_name = fname.replace("_best.pkl", "")
            with open(os.path.join(MODELS_DIR, fname), "rb") as f:
                models[model_name] = pickle.load(f)
    return models


def prepare_test_data(df, feature_info, preprocessor):
    """
    Prepare test data: ensure correct columns exist, handle missing values.
    """
    cat_cols = feature_info["cat_cols"]
    num_cols = feature_info["num_cols"]
    all_feature_cols = num_cols + cat_cols

    # Check for target column
    has_target = "assessmentClass" in df.columns
    y_true = None
    if has_target:
        y_true = df["assessmentClass"]
        df = df.drop(columns=["assessmentClass"])

    # Ensure all expected columns exist
    for col in all_feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only expected feature columns in correct order
    df = df[all_feature_cols]

    # preprocessor handles missing values via SimpleImputer (fitted on training data)
    X_processed = preprocessor.transform(df)

    return X_processed, y_true, has_target


def run_predictions(test_csv_path):
    """Main prediction pipeline."""
    print("=" * 60)
    print("E-Learning Performance Prediction - Test Script")
    print("=" * 60)

    # Load artifacts
    print("\nLoading saved models and preprocessor...")
    preprocessor, label_encoder, feature_info = load_artifacts()
    models = load_models()
    print(f"Loaded models: {list(models.keys())}")

    # Load test data
    print(f"\nLoading test data from: {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    print(f"Test data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Prepare
    X_test, y_true_raw, has_target = prepare_test_data(df, feature_info, preprocessor)

    if has_target and y_true_raw is not None:
        y_true = label_encoder.transform(y_true_raw)
    else:
        y_true = None

    # Predict with each model
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    all_predictions = {}
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        t0 = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - t0

        y_pred_labels = label_encoder.inverse_transform(y_pred)
        all_predictions[model_name] = y_pred_labels

        print(f"Test time: {test_time:.4f}s")
        print(f"Prediction distribution: {dict(pd.Series(y_pred_labels).value_counts())}")

        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {acc:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Save predictions to CSV
    output_df = df.copy() if not has_target else pd.read_csv(test_csv_path)
    for model_name, preds in all_predictions.items():
        output_df[f"predicted_{model_name}"] = preds

    output_path = os.path.join(SCRIPT_DIR, "test_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_classifier.py <path_to_test_csv>")
        print("Example: python test_classifier.py test_data.csv")
        sys.exit(1)

    test_csv_path = sys.argv[1]
    if not os.path.exists(test_csv_path):
        print(f"Error: File not found: {test_csv_path}")
        sys.exit(1)

    run_predictions(test_csv_path)
