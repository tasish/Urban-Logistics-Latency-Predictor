import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

import src.config as config
import src.preprocessing as preprocessing

warnings.filterwarnings('ignore')

def train_model():
    print("Starting Training Pipeline...")
    
    # 1. Load Data
    print(f"Loading data from {config.DATA_PATH}")
    try:
        df = pd.read_csv(config.DATA_PATH)
    except FileNotFoundError:
        print("Data file not found! Please ensure data is in 'data/Food_Delivery_Dataset.csv'")
        return

    # 2. Preprocessing
    print("Cleaning and Engineering features...")
    df = preprocessing.clean_data(df)
    df = preprocessing.feature_engineering(df)
    df_final = preprocessing.get_final_features(df)
    
    X = df_final
    y = df['Time_taken'] # Extracted in feature_engineering

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Hyperparameter Tuning (RandomizedSearchCV)
    print("Tuning Hyperparameters (this may take a moment)...")
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    # Using 3-fold CV for speed while maintaining robustness
    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='r2', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")
    
    # 6. Evaluation
    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"   R2 Score: {r2:.4f}")
    print(f"   MAE:      {mae:.4f} mins")
    print(f"   MSE:      {mse:.4f}")
    
    # 7. Save Artifacts
    print("Saving artifacts...")
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
        
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Training Complete! Models saved to {config.MODELS_DIR}")

if __name__ == "__main__":
    train_model()
