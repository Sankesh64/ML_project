"""
NASA Battery Dataset Preprocessing & Feature Extraction
Processes raw NASA cycle files into a unified ML-ready feature table.
Calculates State of Health (SoH) and Remaining Useful Life (RUL).
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import kagglehub

def get_dataset_dir():
    """Returns path to the downloaded NASA dataset directory."""
    # Check default kagglehub cache or download
    try:
        path = kagglehub.dataset_download("patrickfleith/nasa-battery-dataset")
        clean_dir = os.path.join(path, "cleaned_dataset")
        if os.path.exists(clean_dir):
            return clean_dir
        return path
    except Exception as e:
        print(f"[WARN] kagglehub download check failed: {e}. Checking local cache...")
        cache_path = os.path.expanduser(r"~/.cache/kagglehub/datasets/patrickfleith/nasa-battery-dataset/versions/2/cleaned_dataset")
        if os.path.exists(cache_path):
            return cache_path
        raise FileNotFoundError("Could not locate NASA battery dataset folder.")

def extract_cycle_features(data_dir, filename):
    """Extracts summary time-series features from an individual cycle CSV."""
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        return {}
    
    try:
        df = pd.read_csv(file_path)
        if df.empty or "Time" not in df.columns:
            return {}
        
        duration = float(df["Time"].max() - df["Time"].min())
        v_mean = float(df["Voltage_measured"].mean()) if "Voltage_measured" in df.columns else np.nan
        v_min = float(df["Voltage_measured"].min()) if "Voltage_measured" in df.columns else np.nan
        v_max = float(df["Voltage_measured"].max()) if "Voltage_measured" in df.columns else np.nan
        i_mean = float(df["Current_measured"].mean()) if "Current_measured" in df.columns else np.nan
        t_mean = float(df["Temperature_measured"].mean()) if "Temperature_measured" in df.columns else np.nan
        t_max = float(df["Temperature_measured"].max()) if "Temperature_measured" in df.columns else np.nan
        
        return {
            "duration_s": round(duration, 2),
            "voltage_mean": round(v_mean, 4),
            "voltage_min": round(v_min, 4),
            "voltage_max": round(v_max, 4),
            "current_mean": round(i_mean, 4),
            "temp_mean": round(t_mean, 2),
            "temp_max": round(t_max, 2)
        }
    except Exception as e:
        return {}

def build_dataset(output_csv="data/nasa_battery_cycles.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dataset_dir = get_dataset_dir()
    print(f"[INFO] Using NASA dataset at: {dataset_dir}")
    
    meta_path = os.path.join(dataset_dir, "metadata.csv")
    data_dir = os.path.join(dataset_dir, "data")
    
    print("[INFO] Reading metadata.csv...")
    meta = pd.read_csv(meta_path)
    
    # Filter for discharge cycles only (where capacity degradation is measured)
    discharges = meta[meta["type"] == "discharge"].copy()
    discharges["Capacity"] = pd.to_numeric(discharges["Capacity"], errors="coerce")
    discharges = discharges.dropna(subset=["Capacity"])
    discharges = discharges.sort_values(by=["battery_id", "test_id"]).reset_index(drop=True)
    
    # Assign cycle sequence per battery
    discharges["cycle"] = discharges.groupby("battery_id").cumcount() + 1
    
    # Nominal capacity for NASA 18650 Li-ion cells is 2.0 Ah
    NOMINAL_CAPACITY = 2.0
    discharges["SoH"] = (discharges["Capacity"] / NOMINAL_CAPACITY) * 100
    
    print(f"[INFO] Processing {len(discharges)} discharge cycles across {discharges['battery_id'].nunique()} batteries...")
    
    feature_records = []
    for idx, row in tqdm(discharges.iterrows(), total=len(discharges), desc="Extracting telemetry features"):
        feat = {
            "battery_id": row["battery_id"],
            "cycle": row["cycle"],
            "ambient_temperature": row["ambient_temperature"],
            "Capacity": round(row["Capacity"], 5),
            "SoH": round(row["SoH"], 2),
        }
        
        # Extract cycle time-series features if filename is present
        fn = row["filename"]
        if pd.notna(fn):
            ts_feats = extract_cycle_features(data_dir, fn)
            feat.update(ts_feats)
        
        feature_records.append(feat)
        
    df_feat = pd.DataFrame(feature_records)
    for col in ["ambient_temperature", "Capacity", "SoH", "duration_s", "voltage_mean", "voltage_min", "voltage_max", "current_mean", "temp_mean", "temp_max"]:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
    
    # Calculate RUL (Remaining Useful Life)
    # Standard EoL (End-of-Life) threshold in battery literature is 70% of nominal capacity (1.4 Ah)
    EOL_CAPACITY_THRESHOLD = 1.4
    
    df_feat["RUL"] = 0
    for bat, idxs in df_feat.groupby("battery_id").groups.items():
        sub = df_feat.loc[idxs]
        eol_rows = sub[sub["Capacity"] <= EOL_CAPACITY_THRESHOLD]
        eol_cycle = eol_rows["cycle"].min() if not eol_rows.empty else sub["cycle"].max()
        df_feat.loc[idxs, "RUL"] = np.maximum(0, eol_cycle - sub["cycle"])
    
    # Save processed CSV
    df_feat.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Dataset successfully saved to: {os.path.abspath(output_csv)}")
    print(f"[INFO] Total Records: {len(df_feat)}")
    print(f"[INFO] Batteries: {df_feat['battery_id'].nunique()}")
    print("\n--- First 5 Records ---")
    print(df_feat.head())
    print("\n--- Summary Statistics ---")
    print(df_feat[["cycle", "Capacity", "SoH", "RUL", "voltage_mean", "temp_max"]].describe().round(3))
    
    return df_feat

if __name__ == "__main__":
    build_dataset()
