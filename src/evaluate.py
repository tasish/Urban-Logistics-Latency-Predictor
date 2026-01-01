import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import src.config as config
import src.preprocessing as preprocessing
import warnings
import os

warnings.filterwarnings('ignore')

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score (R2)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    # Save plot
    if not os.path.exists('assets'): os.makedirs('assets')
    plt.savefig('assets/learning_curve.png')
    print("Learning Curve saved to 'assets/learning_curve.png'")

def compare_models(X, y):
    print("\nComparing Models (Random Forest vs XGBoost)...")
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    results = []
    names = []
    
    for name, model in models.items():
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print(f"   {name}: Mean R2 = {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")
        
    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.ylabel('R2 Score')
    plt.savefig('assets/model_comparison.png')
    print("Model Comparison plot saved to 'assets/model_comparison.png'")

def evaluate_pipeline():
    print("Starting Advanced Evaluation...")
    
    # Load & Preprocess
    df = pd.read_csv(config.DATA_PATH)
    df = preprocessing.clean_data(df)
    df = preprocessing.feature_engineering(df)
    df_final = preprocessing.get_final_features(df)
    
    X = df_final
    y = df['Time_taken']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Bias-Variance Analysis (Learning Curve)
    print("\nGenerating Learning Curve for XGBoost...")
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    plot_learning_curve(model, "Learning Curve (XGBoost)", X_scaled, y, cv=5)
    
    # 2. Model Comparison
    compare_models(X_scaled, y)
    
    print("\nEvaluation Complete!")

if __name__ == "__main__":
    evaluate_pipeline()
