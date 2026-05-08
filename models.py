from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results['Linear Regression'] = lr_preds

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest'] = rf_preds

    # Evaluation
    for name, preds in results.items():
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"--- {name} ---")
        print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n")

    return results
    # Plot Results
def plot_results(y_test, predictions_dict):
    plt.figure(figsize=(12, 5))
    
    for i, (name, preds) in enumerate(predictions_dict.items(), 1):
        plt.subplot(1, 2, i)
        plt.scatter(y_test, preds, alpha=0.3, color='blue' if i==1 else 'green')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'{name}: Actual vs Predicted')
        plt.xlabel('Actual Score')
        plt.ylabel('Predicted Score')

    plt.tight_layout()
    plt.show()