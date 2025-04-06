import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. DATA LOADING AND PV SIMULATION (YOUR EXISTING CODE)
# --------------------------------------------------

# Load data
df = pd.read_csv('powerdemand_5min_2021_to_2024_with weather.csv',
                parse_dates=['datetime'],
                index_col='datetime')

# Load calibration
homes_in_group = 10
df['residential_load_kW'] = df['Power demand'] / homes_in_group

# PV Generation setup
site = Location(latitude=28.7041, longitude=77.1025,
               tz='Asia/Kolkata', altitude=216)

pv_config = {
    'system_capacity': 10,
    'tilt': 28,
    'azimuth': 180,
    'module_efficiency': 0.20,  
    'temp_coeff': -0.0035,
    'losses': 0.12
}

# Solar calculations
solar_position = site.get_solarposition(df.index)
clearsky = site.get_clearsky(df.index, model='ineichen')

# POA irradiance and PV output
poa_irrad = pvlib.irradiance.get_total_irradiance(
    surface_tilt=pv_config['tilt'],
    surface_azimuth=pv_config['azimuth'],
    dni=clearsky['dni'],
    ghi=clearsky['ghi'],
    dhi=clearsky['dhi'],
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth']
)

df['pv_power_kw'] = (
    poa_irrad['poa_global'] * 
    pv_config['system_capacity'] * 
    pv_config['module_efficiency'] *
    (1 + pv_config['temp_coeff'] * (df['temp'] - 25)) * 
    (1 - pv_config['losses'])
)

# Nighttime adjustment
df.loc[(df.index.hour < 6) | (df.index.hour >= 18), 'pv_power_kw'] = 0

# --------------------------------------------------
# 2. LOAD FORECASTING MODELS (NEW CODE)
# --------------------------------------------------

def prepare_data_for_forecasting(df, target_col='residential_load_kW', n_steps=24):
    """Create time-series sequences for forecasting"""
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:i+n_steps][[target_col, 'temp']].values)
        y.append(df.iloc[i+n_steps][target_col])
    return np.array(X), np.array(y)

# Prepare features and target
features = ['residential_load_kW', 'temp']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)

# Create sequences
X, y = prepare_data_for_forecasting(df_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
history = lstm_model.fit(X_train, y_train, 
                        epochs=30, 
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1)

# Make predictions
lstm_pred = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(
    np.concatenate([lstm_pred, X_test[:, -1, 1:]], axis=1)
)[:, 0]

# XGBoost Model
X_train_xgb = X_train.reshape(X_train.shape[0], -1)
X_test_xgb = X_test.reshape(X_test.shape[0], -1)

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8
)
xgb_model.fit(X_train_xgb, y_train)
xgb_pred = xgb_model.predict(X_test_xgb)
xgb_pred = scaler.inverse_transform(
    np.concatenate([xgb_pred.reshape(-1,1), X_test[:, -1, 1:]], axis=1)
)[:, 0]

# --------------------------------------------------
# 3. EVALUATION AND RESULTS
# --------------------------------------------------

# Calculate metrics
y_test_actual = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), X_test[:, -1, 1:]], axis=1)
)[:, 0]

def calculate_metrics(actual, pred):
    return {
        'MAE': mean_absolute_error(actual, pred),
        'MAPE': np.mean(np.abs((actual - pred)/actual)) * 100
    }

lstm_metrics = calculate_metrics(y_test_actual, lstm_pred)
xgb_metrics = calculate_metrics(y_test_actual, xgb_pred)

print("LSTM Performance:", lstm_metrics)
print("XGBoost Performance:", xgb_metrics)

# Update net power calculations
test_dates = df.index[-len(y_test):]
df_test = df.loc[test_dates].copy()
df_test['load_forecast_lstm'] = lstm_pred
df_test['load_forecast_xgb'] = xgb_pred
df_test['net_power_lstm'] = df_test['load_forecast_lstm'] - df_test['pv_power_kw']
df_test['net_power_xgb'] = df_test['load_forecast_xgb'] - df_test['pv_power_kw']

# Save and visualize
df_test.to_csv('power_forecast_results.csv')

plt.figure(figsize=(15, 8))
plt.plot(df_test.index, df_test['residential_load_kW'], label='Actual Load', alpha=0.7)
plt.plot(df_test.index, df_test['load_forecast_lstm'], label='LSTM Forecast', linestyle='--')
plt.plot(df_test.index, df_test['load_forecast_xgb'], label='XGBoost Forecast', linestyle='--')
plt.title('Load Forecasting Comparison')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid()
plt.show()