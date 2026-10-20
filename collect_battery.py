"""
Laptop Battery Degradation Predictor - Battery Status Collector CLI
Automatically generates and parses Windows powercfg /batteryreport to monitor battery health.
"""

import os
import re
import sys
import csv
from datetime import datetime
import subprocess

def clean_value(val):
    if not val:
        return None
    # Remove units like 'mWh', commas, or newlines
    clean = re.sub(r'[^\d.]', '', val)
    try:
        if '.' in clean:
            return float(clean)
        return int(clean)
    except ValueError:
        return val

def run_battery_report():
    """Generates the Windows battery report HTML."""
    if sys.platform != "win32":
        print("[ERROR] Battery report generation via powercfg is only supported on Windows.")
        return False
    
    report_path = "battery-report.html"
    print("[STATUS] Running Windows powercfg /batteryreport...")
    try:
        result = subprocess.run(
            ["powercfg", "/batteryreport", "/output", report_path],
            capture_output=True,
            text=True,
            check=True
        )
        if os.path.exists(report_path):
            print(f"[OK] Battery report generated successfully at: {os.path.abspath(report_path)}")
            return True
        else:
            print("[ERROR] Battery report file was not created.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Running powercfg failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error running battery report: {e}")
        return False

def parse_battery_report(source="battery-report.html"):
    """Parses the generated UTF-8/UTF-8-SIG battery report HTML from a file path, file-like object, or raw HTML string."""
    html = ""
    if hasattr(source, "read"):
        content = source.read()
        html = content.decode("utf-8-sig", errors="ignore") if isinstance(content, bytes) else str(content)
    elif isinstance(source, str) and ("<html" in source.lower() or "<!doctype" in source.lower()):
        html = source
    elif isinstance(source, str) and os.path.exists(source):
        try:
            with open(source, "r", encoding="utf-8-sig", errors="ignore") as f:
                html = f.read()
        except Exception:
            try:
                with open(source, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
            except Exception as e:
                print(f"[ERROR] Reading {source}: {e}")
                return None
    else:
        print(f"[ERROR] Invalid battery report source: {source}")
        return None

    def extract_label_val(label):
        pattern = rf'<td><span class="label">{label}</span></td><td>([^<]+)'
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": extract_label_val("NAME"),
        "manufacturer": extract_label_val("MANUFACTURER"),
        "chemistry": extract_label_val("CHEMISTRY"),
        "design_capacity_raw": extract_label_val("DESIGN CAPACITY"),
        "full_charge_capacity_raw": extract_label_val("FULL CHARGE CAPACITY"),
        "cycle_count_raw": extract_label_val("CYCLE COUNT")
    }
    
    # Clean numeric fields
    data["design_capacity"] = clean_value(data["design_capacity_raw"])
    data["full_charge_capacity"] = clean_value(data["full_charge_capacity_raw"])
    data["cycle_count"] = clean_value(data["cycle_count_raw"])
    
    # Calculate SoH
    if isinstance(data["design_capacity"], (int, float)) and isinstance(data["full_charge_capacity"], (int, float)) and data["design_capacity"] > 0:
        data["soh"] = round((data["full_charge_capacity"] / data["design_capacity"]) * 100, 2)
    else:
        data["soh"] = None
        
    return data

def save_to_history(data, csv_path="battery_history.csv"):
    """Appends parsed battery data to a local CSV file."""
    file_exists = os.path.exists(csv_path)
    
    headers = [
        "Timestamp", "Battery Name", "Manufacturer", "Chemistry", 
        "Design Capacity (mWh)", "Full Charge Capacity (mWh)", 
        "Cycle Count", "State of Health (%)"
    ]
    
    row = [
        data["timestamp"],
        data["name"] or "Unknown",
        data["manufacturer"] or "Unknown",
        data["chemistry"] or "Unknown",
        data["design_capacity"] or "",
        data["full_charge_capacity"] or "",
        data["cycle_count"] or "",
        data["soh"] or ""
    ]
    
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)
        print(f"[SAVE] Saved entry to history file: {os.path.abspath(csv_path)}")
        return True
    except Exception as e:
        print(f"[ERROR] Saving to CSV failed: {e}")
        return False

def print_summary(data):
    """Prints a beautiful summary of battery health."""
    print("\n" + "="*50)
    print("         LAPTOP BATTERY HEALTH SUMMARY")
    print("="*50)
    print(f"  Captured At:          {data['timestamp']}")
    print(f"  Battery Name:         {data['name']}")
    print(f"  Manufacturer:         {data['manufacturer']}")
    print(f"  Chemistry:            {data['chemistry']}")
    print(f"  Design Capacity:      {data['design_capacity_raw']}")
    print(f"  Full Charge Capacity: {data['full_charge_capacity_raw']}")
    print(f"  Cycle Count:          {data['cycle_count_raw'] or 'N/A'}")
    
    if data['soh'] is not None:
        soh = data['soh']
        if soh >= 85:
            soh_status = "Excellent"
        elif soh >= 75:
            soh_status = "Good"
        elif soh >= 60:
            soh_status = "Warning (Degraded)"
        else:
            soh_status = "Critical (Replace Recommended)"
            
        print(f"  State of Health (SoH): {soh}% ({soh_status})")
    else:
        print("  State of Health (SoH): Could not calculate")
    print("="*50 + "\n")

def main():
    print("[START] Battery Status Collector CLI starting...")
    success = run_battery_report()
    if not success:
        # Check if battery-report.html already exists as fallback
        if os.path.exists("battery-report.html"):
            print("[WARN] Falling back to existing battery-report.html in workspace...")
        else:
            print("[ERROR] No battery report found or generated. Exiting.")
            sys.exit(1)
            
    data = parse_battery_report()
    if data:
        print_summary(data)
        save_to_history(data)
    else:
        print("[ERROR] Failed to parse battery report data.")

if __name__ == "__main__":
    main()
