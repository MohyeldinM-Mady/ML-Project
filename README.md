# E-Learning Student Performance Predictor

## 📖 Overview
The **E-Learning Student Performance Predictor** is an end-to-end Machine Learning project designed to analyze and predict student outcomes based on their demographic data, course registration details, assessment scores, and virtual learning environment (VLE) interactions. 

The project is divided into two major milestones, tackling the problem from both continuous (Regression) and categorical (Classification) perspectives.

---

## ✨ Features and Functionalities

### 1. Unified Command-Line Interface (`main.py`)
A central hub for interacting with the project. Running `main.py` provides an interactive menu that allows you to easily execute training pipelines or evaluate unseen data for both milestones without manually calling individual scripts.

### 2. Comprehensive Data Preprocessing pipelines
- **Data Merging (`data_preparation.py`, `regression_data_loader.py`)**: Combines multiple complex datasets including student info, assessments, course details, registrations, and large-scale VLE interaction logs.
- **Feature Engineering**: Calculates insightful metrics such as submission delays, registration earliness, and aggregates daily clicks and activity types in the virtual learning environment.
- **Automated Pipelines (`regression_preprocessing.py`, `train_classifiers.py`)**: Employs scikit-learn `Pipeline` and `ColumnTransformer` to handle missing value imputation, standard scaling for numerical features, and ordinal/label encoding for categorical data to prevent data leakage during testing.

### 3. Milestone 1: Regression Pipeline
Predicts a continuous student `score` using regression algorithms.
- **Models Used**: Linear Regression and Random Forest Regressor.
- **Evaluation**: Computes Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared ($R^2$) metrics.
- **Visualizations**: Generates scatter plots comparing Actual vs. Predicted scores (saved to the `plots/` directory).
- **Scripts**: `run_regression.py`, `regression_models.py`

### 4. Milestone 2: Classification Pipeline
Categorizes student performance into descriptive classes (`Fail`, `Good`, `Very Good`, `Excellent`) based on their assessment scores.
- **Models Used**: Decision Tree Classifier, Random Forest Classifier, and Gradient Boosting Classifier.
- **Hyperparameter Tuning**: Automatically trains and searches across a grid of parameters (e.g., `n_estimators`, `max_depth`, `learning_rate`) to find the most optimal configuration.
- **Evaluation**: Uses Accuracy, Classification Reports (Precision, Recall, F1-Score), and records Training/Testing execution times.
- **Visualizations**: Produces detailed bar charts comparing model accuracies, training/test times, and hyperparameter impacts (saved to the `plots/` directory).
- **Scripts**: `train_classifiers.py`

### 5. Testing and Inference on Unseen Data
Dedicated scripts to reliably process and generate predictions for new, unseen datasets.
- Loads saved preprocessors, scalers, and the best-performing trained models from the `saved_models/` directory.
- Drops demographic noise uniformly and prevents data leakage.
- Exports predictions to a CSV file (for classification) or prints metrics to the console.
- **Scripts**: `test_regression.py`, `test_classifier.py`

---

## 📂 Project Structure

```text
ML-Project/
│
├── main.py                       # Unified interactive entry point
│
├── Milestone 1 Data/             # Raw data directory for regression
├── regression_data_loader.py     # Data loading script for regression
├── regression_preprocessing.py   # Preprocessing pipeline for regression
├── run_regression.py             # Main execution script for training regression models
├── regression_models.py          # Model definitions and evaluation logic for regression
├── test_regression.py            # Evaluation script for unseen regression data
│
├── Milestone 2 Data/             # Raw data directory for classification
├── data_preparation.py           # Feature engineering and data preparation for classification
├── train_classifiers.py          # Main execution script for training classification models
├── test_classifier.py            # Evaluation script for unseen classification data
│
├── saved_models/                 # Directory where serialized models, scalers, and preprocessors are stored (.pkl)
├── plots/                        # Directory containing generated graphs and comparison charts
└── processed_dataset.csv         # The final engineered dataset used for Milestone 2
```

---

## 🚀 How to Run

1. **Start the Application**:
   Open your terminal, navigate to the project directory, and run the main script:
   ```bash
   python main.py
   ```

2. **Select an Option**:
   The interactive menu will display the following options:
   ```
   ============================================================
   E-Learning Student Performance Predictor
   ============================================================
   1. Run Milestone 1: Regression Pipeline (Train & Predict)
   2. Run Milestone 2: Classification Pipeline (Train & Predict)
   3. Run Milestone 2: Test Script (Evaluate unseen CLASSIFICATION data)
   4. Run Milestone 1: Test Script (Evaluate unseen REGRESSION data)
   0. Exit
   ------------------------------------------------------------
   ```
   Enter the number corresponding to the action you want to perform. 

3. **Running Evaluations (Options 3 & 4)**:
   If you choose to evaluate unseen data, the script will prompt you to enter the path to the unseen CSV file. Ensure the CSV format matches the training data format (minus the target column, which is optional).
