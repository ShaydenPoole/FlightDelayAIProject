# Flight Delay Prediction
 
Flight delay prediction using machine learning on the U.S. DOT dataset (5.8M+ flights). Compares LightGBM, XGBoost, CatBoost, Decision Tree, and HistGradientBoosting classifiers with feature engineering, hyperparameter tuning, and threshold optimization for F1 maximization.
 
---
 
## Dataset
 
**Source:** [Kaggle — U.S. DOT Flight Delays](https://www.kaggle.com/datasets/usdot/flight-delays)  
**Coverage:** 2015 U.S. domestic flights (~5.8M records)  
**Files:** `flights.csv`, `airlines.csv`, `airports.csv`
 
---
 
## Project Structure
 
```
flight-delay-prediction/
│
├── flight_delays_analysis.ipynb   # Main notebook (EDA + modeling + prediction)
├── saved_models/                  # Trained model pickle files (generated on run)
└── README.md
```
 
---
 
## Workflow
 
### 1. Exploratory Data Analysis
- Arrival and departure delay distributions
- Average delays by airline, day of week, and month
- Top origin cities by delay
- Cancellation rates by airline
- Correlation matrix of delay features
### 2. Feature Engineering
- **Cyclical encodings** — hour, day of week, month (sin/cos)
- **Route features** — origin-destination pair, airline-route interaction
- **Historical statistics** — airline and route-level mean, median, and standard deviation of delays (computed from training data only to prevent leakage)
- **Temporal route patterns** — route-hour and route-day-of-week averages
- **Airport congestion** — flight counts per airport per hour
- **Distance bins** — short / medium / long haul
- **Flight efficiency** — speed in miles per minute
### 3. Models Trained
 
| Model | Tuning Method |
|---|---|
| LightGBM | GridSearchCV |
| XGBoost | Early stopping |
| CatBoost | Early stopping |
| Decision Tree | GridSearchCV |
| HistGradientBoosting | GridSearchCV |
 
- **Target:** Binary classification — delayed (`ARRIVAL_DELAY > 0`) vs. on-time/early
- **Split:** Chronological 70/15/15 (train/validation/test) to preserve temporal dependencies
- **Class imbalance:** Handled via `scale_pos_weight` in LightGBM and XGBoost
- **Threshold optimization:** Per-model F1-maximizing threshold tuned on validation set
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
### 4. Prediction Interface
Input any flight's details (airline, origin, destination, date, time, distance) and get a delay probability from any saved model.
 
---
 
## Setup
 
### Prerequisites
- Python 3.8+
- Kaggle account with API credentials
### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost kaggle
```
 
### Kaggle API setup
Place your `kaggle.json` credentials file in the project root before running the data download cell.  
[How to get your Kaggle API key →](https://www.kaggle.com/docs/api)
 
### Run
Open `flight_delays_analysis.ipynb` and run cells sequentially. The `saved_models/` directory is created automatically during training.
 
> **Note:** Run all training cells before using the prediction interface — the prediction cell loads from saved model files.
 
---
 
## Tech Stack
 
Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · CatBoost · Matplotlib · Seaborn
