# 🔋 Battery Health & RUL AI Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlproject-ycsofma3vkbtnzny3sggpt.streamlit.app/)

> **🚀 [Live Demo → Try it now!](https://mlproject-ycsofma3vkbtnzny3sggpt.streamlit.app/)**

A machine learning system to predict laptop battery degradation patterns, State of Health (SoH), and Remaining Useful Life (RUL) using NASA battery dataset and physics-informed features.

## 📊 Overview

Lithium-ion batteries in modern laptops degrade over time due to electrochemical aging, usage patterns, charge cycles, temperature, and depth of discharge. This project provides a robust, data-driven framework to model, analyze, and predict laptop battery health and degradation trajectories.

The goal is to proactively predict when a battery will reach its End-of-Life (EoL) or fall below acceptable capacity thresholds (typically 70% of design capacity), allowing users and IT administrators to plan replacements efficiently.

## 🔍 Core Features

- **State of Health (SoH) Prediction**: Predicts remaining useful capacity percentage based on charge cycles, thermal profiles, and voltage characteristics.
- **RUL (Remaining Useful Life) Forecast**: Regression models to estimate the remaining cycles before battery capacity drops below the EoL threshold.
- **Interactive Streamlit Dashboard**: Web-based UI for single-cycle prediction, batch inference, laptop battery diagnostics, and model benchmarks.
- **Multiple ML Models**: Gradient Boosting, Random Forest, and custom Gradient Descent Linear Regression — with leave-batteries-out cross-validation.
- **Laptop Battery Diagnostics**: Parse Windows `powercfg /batteryreport` HTML files to assess real-world battery health.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sankesh64/ML_project.git
   cd ML_project
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```

## 📁 Repository Structure

```
├── app.py                     # Streamlit web application (main entry point)
├── train_models.py            # Model training pipeline (Gradient Boosting, Random Forest, GD Linear)
├── prepare_nasa_dataset.py    # NASA battery dataset preparation & feature engineering
├── collect_battery.py         # Windows battery report parser (powercfg /batteryreport)
├── lap.py                     # Core battery prediction logic
├── models/                    # Pre-trained model files (.joblib)
├── data/                      # NASA battery cycle dataset
├── visualizations/            # Generated plots and charts
├── .streamlit/                # Streamlit configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🏆 Model Performance (Unseen Battery Cells)

| Model | SoH R² | SoH MAE | RUL R² | RUL MAE |
|-------|--------|---------|--------|---------|
| **Gradient Boosting** | 0.887 | 2.14% | 0.892 | 8.45 cycles |
| Random Forest | 0.865 | 2.39% | 0.872 | 9.30 cycles |
| Linear (Gradient Descent) | 0.741 | 3.82% | 0.768 | 14.80 cycles |

## 📈 Roadmap

- [x] Integrate battery status collector CLI (`powercfg /batteryreport` on Windows)
- [x] Implement Gradient Boosting and Random Forest models for degradation prediction
- [x] Build interactive Streamlit dashboard for real-time battery health tracking
- [x] Deploy to Streamlit Community Cloud
- [ ] Add LSTM / deep learning model for sequential degradation prediction
- [ ] Add model registry and experiment tracking (MLflow)

## 🤝 Contributing

Contributions are welcome! Please feel free to open a Pull Request or create an Issue if you have feature suggestions, bug reports, or performance improvements.

## 📄 License

This project is licensed under the MIT License.
