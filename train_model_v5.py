import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from pandas.tseries.holiday import USFederalHolidayCalendar
import json

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
WEATHER_PATH = 'weather_data_hourly_master.csv'

print("Initializing V5 Training Pipeline (Accuracy Optimized)...")

# Try loading V5, fallback to V6
try:
    df_flights = pd.read_csv('flight_ops_master_data_v6.csv', low_memory=False)
except FileNotFoundError:
    df_flights = pd.read_csv('flight_ops_master_data_v6.csv', low_memory=False)

df_weather = pd.read_csv(WEATHER_PATH)

# ==========================================
# 2. MERGE & BASIC ENGINEERING
# ==========================================
print("Aligning & Merging Data...")
df_flights['FL_DATE'] = pd.to_datetime(df_flights['FL_DATE']) 
df_flights['Join_Hour'] = (df_flights['CRS_DEP_TIME'] // 100).astype(int)
# Create a pure Date column for joining
df_flights['Join_Date'] = df_flights['FL_DATE'].dt.date 

df_weather['Weather_Timestamp'] = pd.to_datetime(df_weather['Weather_Timestamp'])
# RENAME to 'Weather_Date' to avoid collision with 'FL_DATE'
df_weather['Weather_Date'] = df_weather['Weather_Timestamp'].dt.date
df_weather['Join_Hour'] = df_weather['Weather_Timestamp'].dt.hour

# Merge using the new name for the right side
df = pd.merge(df_flights, df_weather, 
              left_on=['Join_Date', 'Join_Hour', 'ORIGIN'],
              right_on=['Weather_Date', 'Join_Hour', 'ORIGIN'], # Changed from FL_DATE to Weather_Date
              how='left')

# Fill Gaps
df['temp'] = df['temp'].fillna(df['temp'].mean())
df['wspd'] = df['wspd'].fillna(df['wspd'].mean())
df['prcp'] = df['prcp'].fillna(0)

# Snow Proxy
snow_codes = [14, 15, 16, 21, 22]
df['Is_Snowing'] = 0
if 'coco' in df.columns:
    df['coco'] = df['coco'].fillna(0)
    df.loc[df['coco'].isin(snow_codes), 'Is_Snowing'] = 1
mask_freezing_rain = (df['temp'] < 0) & (df['prcp'] > 0)
df.loc[mask_freezing_rain, 'Is_Snowing'] = 1

# ==========================================
# 3. ADVANCED FEATURE ENGINEERING (NEW)
# ==========================================
print("Applying Advanced Features (Cyclical & Holidays)...")

# A. Cyclical Encoding (Hour & Month)
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# B. Holiday Flag
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df['FL_DATE'].min(), end=df['FL_DATE'].max())
df['Is_Holiday'] = df['FL_DATE'].isin(holidays).astype(int)

# ==========================================
# 4. PANDEMIC PURGE & SPLIT
# ==========================================
print("Filtering Training Data (Dropping 2020-2021)...")

# Define Features (Removed raw Month/Hour, Added Sin/Cos/Holiday)
features = [
    'Route_Risk_Score', 'Is_Pandemic', 'Congestion', 'DISTANCE',
    'temp', 'wspd', 'prcp', 'Is_Snowing',
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 'Is_Holiday'
]
target = 'Disruption_Severity'

# Filter: Remove Pandemic Years from the Dataset entirely for modeling
# We keep 2019, 2022, 2023. We drop 2020, 2021.
normal_ops_mask = ~df['Year'].isin([2020, 2021])
df_model = df[normal_ops_mask].copy()

print(f"Data Reduced from {len(df):,} to {len(df_model):,} rows (Normal Ops only).")

# One-Hot Encoding
print("Encoding Features...")
X = pd.get_dummies(df_model[features + ['AIRLINE', 'ORIGIN']], columns=['AIRLINE', 'ORIGIN'])
y = df_model[target]

# Train/Test Split (Cutoff: Jan 1, 2023)
split_date = pd.Timestamp('2023-01-01')
train_mask = df_model['FL_DATE'] < split_date
test_mask = df_model['FL_DATE'] >= split_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"Training Set: {len(X_train):,} rows (2019 + 2022)")
print(f"Testing Set: {len(X_test):,} rows (2023)")

# ==========================================
# 5. MODEL TRAINING (MANUAL WEIGHTS)
# ==========================================
print("Training XGBoost (Manual Weights)...")

# Custom Weights: Penalize Cancellations (3) heavily, but let On-Time (0) represent itself
# 0: 1x, 1: 2x, 2: 5x, 3: 15x
weights_dict = {0: 1, 1: 2, 2: 5, 3: 15}
sample_weights = y_train.map(weights_dict)

model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=4, 
    n_estimators=150, 
    max_depth=6, 
    learning_rate=0.1, 
    n_jobs=-1, 
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# ==========================================
# 6. METRICS & ARTIFACTS
# ==========================================
print("\nCalculating Performance Metrics...")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['On-Time', 'Moderate', 'Severe', 'Cancelled']))

# A. Metrics for Dashboard
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
    "test_samples": len(y_test)
}

# B. Confusion Matrix
cm = confusion_matrix(y_test, y_pred).tolist()

# C. Feature Importance
importances = dict(zip(X.columns, model.feature_importances_.tolist()))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
importance_dict = dict(sorted_imp)

print("Saving Artifacts...")

# 1. Model
model.save_model("disruption_model_v5.json")

# 2. Schema
with open('model_features_v5.json', 'w') as f:
    json.dump(list(X.columns), f)

# 3. Class Map
class_map = {0: 'On-Time', 1: 'Moderate', 2: 'Severe', 3: 'Cancelled'}
with open('class_mapping_v5.json', 'w') as f:
    json.dump(class_map, f)

# 4. Dashboard Intelligence Files
with open('model_metrics_v5.json', 'w') as f:
    json.dump(metrics, f)

with open('confusion_matrix_v5.json', 'w') as f:
    json.dump(cm, f)

with open('feature_importance_v5.json', 'w') as f:
    json.dump(importance_dict, f)

print("✅ V5 Training Complete. Accuracy Optimized.")