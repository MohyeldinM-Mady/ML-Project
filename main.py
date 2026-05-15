import sys
import subprocess

def main():
    print("="*60)
    print("E-Learning Student Performance Predictor")
    print("="*60)
    print("1. Run Milestone 1: Regression Pipeline (Train & Predict)")
    print("2. Run Milestone 2: Classification Pipeline (Train & Predict)")
    print("3. Run Milestone 2: Test Script (Evaluate unseen CLASSIFICATION data)")
    print("4. Run Milestone 1: Test Script (Evaluate unseen REGRESSION data)")
    print("0. Exit")
    print("-" * 60)
    
    choice = input("Enter your choice (0-4): ")
    
    if choice == '1':
        print("\n[Starting Regression Pipeline...]")
        subprocess.run([sys.executable, "run_regression.py"])
        
    elif choice == '2':
        print("\n[Starting Classification Training Pipeline...]")
        subprocess.run([sys.executable, "train_classifiers.py"])
        
    elif choice == '3':
        test_file = input("Enter the path to the unseen CLASSIFICATION CSV: ")
        print(f"\n[Running Classification Evaluation on {test_file}...]")
        subprocess.run([sys.executable, "test_classifier.py", test_file])
        
    elif choice == '4':
        test_file = input("Enter the path to the unseen REGRESSION CSV: ")
        print(f"\n[Running Regression Evaluation on {test_file}...]")
        subprocess.run([sys.executable, "test_regression.py", test_file])
        
    elif choice == '0':
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()