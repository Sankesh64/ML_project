import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD REAL NASA DATASET OR FALLBACK TO MOCK
# ==========================================
data_file = "data/nasa_battery_cycles.csv"

if os.path.exists(data_file):
    print(f"[DATA] Loading real NASA Battery Dataset from: {data_file}")
    df = pd.read_csv(data_file)
    # Filter for standard NASA cell B0005 (or use all cells)
    cell_data = df[df["battery_id"] == "B0005"].copy()
    if cell_data.empty:
        cell_data = df.copy()
        
    cycles = cell_data["cycle"].values
    soh = cell_data["SoH"].values
    data_label = "Real NASA Battery (B0005)"
    print(f"[DATA] Loaded {len(cycles)} degradation cycles for {data_label}")
else:
    print("[DATA] Real dataset not found. Generating mock degradation data...")
    np.random.seed(42)
    cycles = np.arange(1, 501)
    soh = 100 - (cycles * 0.06) + np.random.normal(0, 2, 500)
    data_label = "Simulated Mock Data"

# Reshape arrays for matrix math
X = cycles.reshape(-1, 1)
y = soh.reshape(-1, 1)

# Feature Scaling: CRITICAL for Gradient Descent. 
# Without scaling cycles, the gradients explode and the model fails.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. HYPERPARAMETERS
# ==========================================
learning_rate = 0.08  # Step size down error surface
epochs = 100          # Number of passes over dataset

# ==========================================
# 3. INITIALIZE PARAMETERS (w and b)
# ==========================================
np.random.seed(42)
w = np.random.randn(1, 1) * 0.01  # Weight (Slope)
b = np.mean(y).reshape(1, 1)      # Bias (Initialized near mean SoH)

loss_history = []
print(f"Starting Training on {data_label} for {epochs} epochs...")

# ==========================================
# 4. GRADIENT DESCENT LOOP
# ==========================================
for epoch in range(epochs):
    # Step A: Make a prediction with current w and b
    y_pred = np.dot(X_scaled, w) + b
    
    # Step B: Calculate Mean Squared Error (MSE) Loss
    error = y_pred - y
    mse = np.mean(error ** 2)
    loss_history.append(mse)
    
    # Step C: Calculate Gradients
    dw = (2 / len(X_scaled)) * np.dot(X_scaled.T, error)
    db = (2 / len(X_scaled)) * np.sum(error)
    
    # Step D: Update Parameters
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}: Loss (MSE) = {mse:.3f}")

final_loss = loss_history[-1]
mae = np.mean(np.abs(y_pred - y))
print(f"Training Complete! Final MSE: {final_loss:.3f} | MAE: {mae:.2f}% SoH")

# ==========================================
# 5. PLOTTING THE RESULTS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Loss Curve
ax1.plot(range(1, epochs + 1), loss_history, color='red', linewidth=2)
ax1.set_title('Gradient Descent Loss Curve (MSE over Epochs)', fontsize=12)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')
ax1.grid(True, linestyle=":", alpha=0.6)

# Graph 2: Learned Fit Line
final_predictions = np.dot(X_scaled, w) + b
ax2.scatter(X, y, color='blue', alpha=0.6, s=18, label=f'Actual Data: {data_label}')
ax2.plot(X, final_predictions, color='green', linewidth=2.5, label='Learned Best Fit Line (GD)')
ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='EoL Critical Threshold (70% SoH)')
ax2.set_title('Linear Regression: Cycle Count vs State of Health', fontsize=12)
ax2.set_xlabel('Cycle Count')
ax2.set_ylabel('State of Health (%)')
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
plot_path = "visualizations/lap_gradient_descent.png"
plt.savefig(plot_path, dpi=180)
print(f"[PLOT] Saved plot to: {os.path.abspath(plot_path)}")
plt.close()