✈️ Flight Arrival Time Change Prediction

A machine learning system for predicting flight arrival delays and delay duration using operational flight data, airport infrastructure information, and historical weather conditions. The project combines classification and regression models to provide both early delay detection and delay severity estimation.

📌 Project Overview

Flight delays create significant operational challenges for airlines and airports. This project leverages machine learning to predict:

Whether a flight will arrive delayed or on time.
The expected delay duration (in minutes).

The system integrates:

Flight operational data (ETA/ETD updates)
Airport infrastructure information
Historical weather conditions
Temporal and operational features

The final solution uses ensemble machine learning techniques including XGBoost and LightGBM, along with SHAP explainability for model interpretation.

🎯 Objectives
Predict flight arrival delays before arrival.
Estimate delay duration in minutes.
Analyze the impact of weather and operational factors.
Build a production-oriented predictive pipeline.
Provide interpretable predictions using SHAP.
📊 Dataset
Flight Operations Dataset
2.7M+ raw flight records
Airline operational updates (ETA/ETD)
Flight schedules
Arrival and departure variance metrics
Airport Information Dataset
4,600+ airport records
Runway information
Geographic attributes
Airport infrastructure characteristics
Weather Dataset

Historical DFW Airport weather data (2021–2025) collected using Open-Meteo API:

Temperature
Humidity
Pressure
Wind Speed
Wind Gusts
Rainfall
Cloud Cover
Weather Severity Indicators

Final modeling dataset:

1,122,809 unique flights
14+ engineered features
🛠️ Tech Stack
Programming Language
Python
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Machine Learning
Scikit-Learn
XGBoost
LightGBM
Explainability
SHAP
🔄 Machine Learning Pipeline
Data Processing
Duplicate flight removal
ETD record filtering
CYCLE record fallback handling
Weather integration
Missing value treatment
Outlier handling (Winsorization)
Feature Engineering
Temporal features
Cyclical encoding (sin/cos)
Historical delay statistics
ETD volatility metrics
Weather severity indicators
Peak-hour interaction features
Leakage Prevention
Removal of post-arrival information
Correlation-based leakage detection
Temporal train/test/validation splitting
📈 Models Implemented
Classification Models
Logistic Regression
Random Forest
LightGBM Classifier
XGBoost Classifier
Regression Models
LightGBM Regressor
XGBoost Regressor
🏆 Classification Results
Model	Accuracy	Recall	F1 Score	AUC-ROC
XGBoost	85.56%	69.58%	78.51%	0.8745
LightGBM	85.54%	69.59%	78.50%	0.8736
Random Forest	85.68%	68.26%	78.33%	0.8693
Logistic Regression	85.68%	68.31%	78.35%	0.8566
Best Classification Model

✅ XGBoost Classifier

Validation Accuracy: 83.65%
Validation AUC-ROC: 0.8712
Precision: 91%
Recall: 70%
📉 Regression Results
Model	MAE (min)	RMSE (min)	R²
XGBoost	9.72	25.58	0.731
LightGBM	9.70	25.61	0.729
Best Regression Model

✅ XGBoost Regressor

MAE: 11.38 minutes
RMSE: 27.28 minutes
R² Score: 0.723
🔍 SHAP Explainability

SHAP analysis identified the most influential features:

MINS_TO_SCHD_DEP_QTY
TTL_POST_QTY
ETD Volatility
Flight Historical Delay Rate
Wind Speed
Pressure
Humidity
Precipitation

Key insight:

Operational signals were stronger predictors of delays than weather alone, while adverse weather conditions significantly increased delay probability.

📂 Project Structure
Flight-Arrival-Time-Change-Prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   ├── Modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   ├── evaluate.py
│
├── models/
│   ├── xgboost_classifier.pkl
│   ├── xgboost_regressor.pkl
│
├── reports/
│   ├── figures/
│   ├── shap_analysis/
│
├── requirements.txt
├── README.md
└── LICENSE
🚀 Key Achievements
Processed 2.7M+ flight records
Built models on 1.12M+ unique flights
Achieved 0.87 AUC-ROC for delay classification
Achieved 73% R² for delay duration prediction
Integrated weather + operational signals
Implemented SHAP explainability
Developed a scalable and production-oriented ML pipeline
📚 Future Improvements
Real-time flight delay prediction
Integration with live weather APIs
Deep learning architectures (LSTM/Transformer)
Multi-airport deployment
Model serving using FastAPI and Docker
MLOps deployment with cloud infrastructure
