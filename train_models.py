"""
Model Training Pipeline for NASA Battery Degradation & RUL Prediction
Trains and compares:
1. Linear Regression via Gradient Descent (Enhanced lap.py module)
2. Random Forest Regressor
3. Gradient Boosting Regressor
Evaluates on unseen battery cells and generates visualization plots.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# 1. Custom Gradient Descent Linear Regression (Matches lap.py architecture)
# ==============================================================================
class GradientDescentRegressor:
    def __init__(self, learning_rate=0.05, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, n_features = X_scaled.shape
        
        np.random.seed(42)
        self.w = np.random.randn(n_features, 1) * 0.01
        self.b = np.zeros((1, 1))
        self.loss_history = []
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X_scaled, self.w) + self.b
            error = y_pred - y
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)
            
            # Gradients
            dw = (2.0 / n_samples) * np.dot(X_scaled.T, error)
            db = (2.0 / n_samples) * np.sum(error)
            
            # Parameter update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
        return self
        
    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before calling 'predict'.")
        X_scaled = self.scaler.transform(X)
        return (np.dot(X_scaled, self.w) + self.b).ravel()

# ==============================================================================
# 2. Main Training & Evaluation Routine
# ==============================================================================
def train_battery_models(data_path="data/nasa_battery_cycles.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found. Running prepare_nasa_dataset.py first...")
        from prepare_nasa_dataset import build_dataset
        build_dataset(data_path)
        
    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop rows with missing values
    feature_cols = ["cycle", "ambient_temperature", "voltage_mean", "voltage_min", "temp_max", "duration_s"]
    df = df.dropna(subset=feature_cols + ["SoH", "RUL"]).reset_index(drop=True)
    
    print(f"[INFO] Usable clean records: {len(df)} across {df['battery_id'].nunique()} batteries")
    
    # --------------------------------------------------------------------------
    # Train / Test Split by Battery (Leave-Batteries-Out Validation)
    # Using famous test cells B0006 and B0018 for testing, others for training
    # --------------------------------------------------------------------------
    test_batteries = ["B0006", "B0018"]
    # If not present, take last 20% of unique batteries
    all_batteries = df["battery_id"].unique().tolist()
    if not all(b in all_batteries for b in test_batteries):
        test_batteries = all_batteries[-max(2, len(all_batteries)//5):]
        
    train_df = df[~df["battery_id"].isin(test_batteries)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_batteries)].reset_index(drop=True)
    
    n_train_bats = len(np.unique(train_df["battery_id"]))
    print(f"[INFO] Train batteries: {n_train_bats} ({len(train_df)} cycles)")
    print(f"[INFO] Test batteries:  {test_batteries} ({len(test_df)} cycles)")
    
    X_train = train_df[feature_cols]
    y_train_soh = train_df["SoH"]
    y_train_rul = train_df["RUL"]
    
    X_test = test_df[feature_cols]
    y_test_soh = test_df["SoH"]
    y_test_rul = test_df["RUL"]
    
    # ==========================================================================
    # TASK 1: State of Health (SoH %) Degradation Prediction
    # ==========================================================================
    print("\n" + "="*60)
    print(" TASK 1: STATE OF HEALTH (SoH %) MODEL TRAINING & BENCHMARK")
    print("="*60)
    
    models_soh = {
        "Gradient Descent (Linear)": GradientDescentRegressor(learning_rate=0.05, epochs=300),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    metrics_soh = []
    predictions_soh = {}
    
    for name, model in models_soh.items():
        print(f"[TRAIN] Fitting {name} for SoH...")
        model.fit(X_train, y_train_soh)
        preds = model.predict(X_test)
        predictions_soh[name] = preds
        
        mse = mean_squared_error(y_test_soh, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_soh, preds)
        r2 = r2_score(y_test_soh, preds)
        
        metrics_soh.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "MAE (%)": round(mae, 2),
            "RMSE (%)": round(rmse, 2)
        })
        
        # Save model artifact
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/soh_{slug}.joblib")
        
    df_metrics_soh = pd.DataFrame(metrics_soh)
    print("\n--- SoH Evaluation Results on Unseen Test Batteries ---")
    print(df_metrics_soh.to_string(index=False))
    
    # ==========================================================================
    # TASK 2: Remaining Useful Life (RUL) Prediction
    # ==========================================================================
    print("\n" + "="*60)
    print(" TASK 2: REMAINING USEFUL LIFE (RUL) MODEL TRAINING & BENCHMARK")
    print("="*60)
    
    # For RUL, SoH is also an input feature
    feature_cols_rul = feature_cols + ["SoH"]
    X_train_rul = train_df[feature_cols_rul]
    X_test_rul = test_df[feature_cols_rul]
    
    models_rul = {
        "Gradient Descent (Linear)": GradientDescentRegressor(learning_rate=0.03, epochs=300),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    metrics_rul = []
    predictions_rul = {}
    
    for name, model in models_rul.items():
        print(f"[TRAIN] Fitting {name} for RUL...")
        model.fit(X_train_rul, y_train_rul)
        preds = model.predict(X_test_rul)
        predictions_rul[name] = preds
        
        mse = mean_squared_error(y_test_rul, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_rul, preds)
        r2 = r2_score(y_test_rul, preds)
        
        metrics_rul.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "MAE (Cycles)": round(mae, 2),
            "RMSE (Cycles)": round(rmse, 2)
        })
        
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/rul_{slug}.joblib")
        
    df_metrics_rul = pd.DataFrame(metrics_rul)
    print("\n--- RUL Evaluation Results on Unseen Test Batteries ---")
    print(df_metrics_rul.to_string(index=False))
    
    # ==========================================================================
    # 3. Visualizations & Figures Generation
    # ==========================================================================
    print("\n[INFO] Generating publication-quality evaluation visualizations...")
    
    # Plot 1: SoH Degradation Curve on Test Battery
    plt.figure(figsize=(10, 6))
    primary_test_bat = test_batteries[0]
    mask_series = test_df["battery_id"] == primary_test_bat
    sub_test = test_df.loc[mask_series].reset_index(drop=True)
    mask = mask_series.values
    
    plt.scatter(sub_test["cycle"], sub_test["SoH"], color="black", s=25, label=f"Actual Battery ({primary_test_bat})", alpha=0.7)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for (m_name, m_preds), c in zip(predictions_soh.items(), colors):
        plt.plot(sub_test["cycle"], m_preds[mask], label=f"{m_name} (Pred)", linewidth=2, color=c)
        
    plt.axhline(y=70, color="red", linestyle="--", alpha=0.7, label="EoL Threshold (70% SoH)")
    plt.title(f"Battery SoH Degradation Forecast: Actual vs Predictions ({primary_test_bat})", fontsize=13, fontweight="bold")
    plt.xlabel("Cycle Count", fontsize=11)
    plt.ylabel("State of Health (%)", fontsize=11)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("visualizations/soh_prediction_curves.png", dpi=180)
    plt.close()
    
    # Plot 2: RUL Forecast Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sub_test["cycle"], sub_test["RUL"], color="black", linewidth=2.5, label=f"Ground Truth RUL ({primary_test_bat})")
    for (m_name, m_preds), c in zip(predictions_rul.items(), colors):
        plt.plot(sub_test["cycle"], m_preds[mask], label=f"{m_name}", linewidth=2, linestyle="--", color=c)
        
    plt.title(f"Remaining Useful Life (RUL) Forecast: Actual vs Predicted ({primary_test_bat})", fontsize=13, fontweight="bold")
    plt.xlabel("Cycle Count", fontsize=11)
    plt.ylabel("Remaining Useful Cycles", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("visualizations/rul_prediction_curves.png", dpi=180)
    plt.close()
    
    # Plot 3: Gradient Descent Loss Curves
    gd_soh = models_soh["Gradient Descent (Linear)"]
    gd_rul = models_rul["Gradient Descent (Linear)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(gd_soh.loss_history, color="#d62728", linewidth=2)
    ax1.set_title("Gradient Descent MSE Loss: SoH Task")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Mean Squared Error")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.plot(gd_rul.loss_history, color="#9467bd", linewidth=2)
    ax2.set_title("Gradient Descent MSE Loss: RUL Task")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Mean Squared Error")
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("visualizations/gradient_descent_loss.png", dpi=180)
    plt.close()
    
    # Plot 4: Random Forest Feature Importance
    rf_model = models_soh["Random Forest Regressor"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices], color="#2ca02c", align="center")
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.title("Random Forest: Feature Importance for Battery Degradation")
    plt.tight_layout()
    plt.savefig("visualizations/feature_importance.png", dpi=180)
    plt.close()
    
    print("\n[SUCCESS] All plots saved to 'visualizations/' directory:")
    print(" - visualizations/soh_prediction_curves.png")
    print(" - visualizations/rul_prediction_curves.png")
    print(" - visualizations/gradient_descent_loss.png")
    print(" - visualizations/feature_importance.png")
    print("[SUCCESS] All trained models saved to 'models/' directory.")

if __name__ == "__main__":
    train_battery_models()
