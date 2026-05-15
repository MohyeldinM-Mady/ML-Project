import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ============================================================
#                       CONFIGURATION
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
#                    LOAD & PREPARE DATA
# ============================================================
def load_and_prepare():
    print("=" * 60, flush=True)
    print("STEP 1: Loading and Preparing Data", flush=True)
    print("=" * 60, flush=True)

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_dataset.csv"))
    print(f"Dataset shape: {df.shape}")

    # Target
    TARGET = "assessmentClass"
    y_raw = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Identify column types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {list(le.classes_)}")
    print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # Build preprocessor
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Fit preprocessor on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Save preprocessor, label encoder, & column info
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump({"cat_cols": cat_cols, "num_cols": num_cols}, f)

    print("Preprocessor & LabelEncoder saved.")
    return X_train_proc, X_test_proc, y_train, y_test, le


# ============================================================
#                DEFINE MODELS & HYPERPARAMETERS
# ============================================================
def get_model_configs():
    
    configs = {}

    # --- Random Forest ---
    configs["RandomForest"] = {
        "base_params": {"random_state": RANDOM_STATE, "n_jobs": -1},
        "hyperparam_1": {
            "name": "n_estimators",
            "values": [50, 100, 200],
            "fixed_other": {"max_depth": 15},
        },
        "hyperparam_2": {
            "name": "max_depth",
            "values": [5, 15, 30],
            "fixed_other": {"n_estimators": 100},
        },
        "model_class": RandomForestClassifier,
    }

    # --- Gradient Boosting ---
    configs["GradientBoosting"] = {
        "base_params": {"random_state": RANDOM_STATE},
        "hyperparam_1": {
            "name": "n_estimators",
            "values": [50, 100, 200],
            "fixed_other": {"learning_rate": 0.1, "max_depth": 3},
        },
        "hyperparam_2": {
            "name": "learning_rate",
            "values": [0.01, 0.1, 0.3],
            "fixed_other": {"n_estimators": 100, "max_depth": 3},
        },
        "model_class": GradientBoostingClassifier,
    }

    # --- Decision Tree ---
    configs["DecisionTree"] = {
        "base_params": {"random_state": RANDOM_STATE},
        "hyperparam_1": {
            "name": "max_depth",
            "values": [5, 15, 30],
            "fixed_other": {"min_samples_split": 5},
        },
        "hyperparam_2": {
            "name": "min_samples_split",
            "values": [2, 5, 10],
            "fixed_other": {"max_depth": 15},
        },
        "model_class": DecisionTreeClassifier,
    }

    return configs


# ============================================================
#                      TRAIN & EVALUATE
# ============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    print("\n" + "=" * 60)
    print("STEP 2: Training & Evaluating Classifiers")
    print("=" * 60)

    configs = get_model_configs()
    all_results = []          # summary bar charts
    hyperparam_results = {}   # hyperparameter analysis
    best_models = {}          # best model per algorithm

    for model_name, config in configs.items():
        print(f"\n{'-' * 50}", flush=True)
        print(f"  MODEL: {model_name}", flush=True)
        print(f"{'-' * 50}", flush=True)

        model_class = config["model_class"]
        base_params = config["base_params"]
        best_acc = 0
        best_model = None
        hp_results = {}

        for hp_key in ["hyperparam_1", "hyperparam_2"]:
            hp_config = config[hp_key]
            hp_name = hp_config["name"]
            hp_values = hp_config["values"]
            fixed = hp_config["fixed_other"]

            hp_results[hp_name] = {"values": [], "accuracies": [], "train_times": [], "test_times": []}

            print(f"\n  Varying '{hp_name}': {hp_values}  (fixed: {fixed})", flush=True)

            for val in hp_values:
                params = {**base_params, **fixed, hp_name: val}
                model = model_class(**params)

                # Train
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                # Test
                t0 = time.time()
                y_pred = model.predict(X_test)
                test_time = time.time() - t0

                acc = accuracy_score(y_test, y_pred)
                hp_results[hp_name]["values"].append(val)
                hp_results[hp_name]["accuracies"].append(acc)
                hp_results[hp_name]["train_times"].append(train_time)
                hp_results[hp_name]["test_times"].append(test_time)

                print(f"    {hp_name}={str(val):>10} | Acc: {acc:.4f} | Train: {train_time:.2f}s | Test: {test_time:.2f}s", flush=True)

                if acc > best_acc:
                    best_acc = acc
                    best_model = model

        # Save best model
        best_models[model_name] = best_model
        model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"\n  Best {model_name} accuracy: {best_acc:.4f} (saved)")

        # Get final metrics on best model
        y_pred = best_model.predict(X_test)
        t0 = time.time()
        best_model.predict(X_test)
        final_test_time = time.time() - t0

        all_results.append({
            "Model": model_name,
            "Accuracy": best_acc,
            "Train Time (s)": max(hp_results[list(hp_results.keys())[0]]["train_times"]),
            "Test Time (s)": final_test_time,
        })

        # Classification report
        print(f"\n  Classification Report ({model_name}):")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        hyperparam_results[model_name] = hp_results

    return all_results, hyperparam_results, best_models


# ============================================================
#                       GENERATE PLOTS
# ============================================================
def generate_plots(all_results, hyperparam_results):
    print("\n" + "=" * 60)
    print("STEP 3: Generating Plots")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # --- Classification Accuracy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df["Model"], results_df["Accuracy"], color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, results_df["Accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Classification Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()

    # --- Training Time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df["Model"], results_df["Train Time (s)"], color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, results_df["Train Time (s)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Total Training Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_time_comparison.png"), dpi=150)
    plt.close()

    # --- Test Time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df["Model"], results_df["Test Time (s)"], color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, results_df["Test Time (s)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}s", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Test Time (seconds)", fontsize=12)
    ax.set_title("Total Test Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "test_time_comparison.png"), dpi=150)
    plt.close()

    # --- Hyperparameter Tuning Plots ---
    for model_name, hp_data in hyperparam_results.items():
        for hp_name, data in hp_data.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            x_labels = [str(v) for v in data["values"]]
            bars = ax.bar(x_labels, data["accuracies"], color="#7E57C2", edgecolor="black", width=0.4)
            for bar, val in zip(bars, data["accuracies"]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=10)
            ax.set_xlabel(hp_name, fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_title(f"{model_name}: Effect of {hp_name}", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_{hp_name}_tuning.png"), dpi=150)
            plt.close()

    print(f"All plots saved to: {PLOTS_DIR}")


# ============================================================
#                            MAIN
# ============================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_and_prepare()
    all_results, hp_results, best_models = train_and_evaluate(X_train, X_test, y_train, y_test, le)
    generate_plots(all_results, hp_results)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Plots saved in:  {PLOTS_DIR}")
    print("\nSummary:")
    for r in all_results:
        print(f"  {r['Model']:25s} | Accuracy: {r['Accuracy']:.4f} | Train: {r['Train Time (s)']:.2f}s | Test: {r['Test Time (s)']:.4f}s")
