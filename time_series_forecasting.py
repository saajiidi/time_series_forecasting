import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Set style for plots
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (14, 7)

# Load the Air Passengers dataset
print("Loading Air Passengers dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
data.columns = ['Passengers']

# Plot the time series
print("\nPlotting the time series...")
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Passengers'])
plt.title('Monthly Airline Passengers (1949-1960)')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()

# Check for stationarity
print("\nChecking for stationarity...")
result = adfuller(data['Passengers'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

# Make the time series stationary by differencing
data['Passengers_diff'] = data['Passengers'].diff().dropna()

# Plot ACF and PACF to identify AR and MA terms
print("\nPlotting ACF and PACF...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
plot_acf(data['Passengers_diff'].dropna(), lags=40, ax=ax1)
plot_pacf(data['Passengers_diff'].dropna(), lags=40, ax=ax2)
plt.tight_layout()
plt.show()

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['Passengers'][:train_size], data['Passengers'][train_size:]
print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

# ARIMA Model
print("\nTraining ARIMA model...")
arima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
arima_result = arima_model.fit(disp=False)
print(arima_result.summary())

# Make predictions with ARIMA
arima_forecast = arima_result.forecast(steps=len(test))

# Calculate ARIMA metrics
arima_rmse = sqrt(mean_squared_error(test, arima_forecast))
arima_mae = mean_absolute_error(test, arima_forecast)
print(f"\nARIMA RMSE: {arima_rmse:.2f}")
print(f"ARIMA MAE: {arima_mae:.2f}")

# Prepare data for Prophet
print("\nPreparing data for Prophet...")
df_prophet = data[['Passengers']].reset_index()
df_prophet.columns = ['ds', 'y']

# Split data for Prophet
prophet_train = df_prophet.iloc[:train_size]
prophet_test = df_prophet.iloc[train_size:]

# Train Prophet model
print("Training Prophet model...")
prophet_model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
prophet_model.fit(prophet_train)

# Create future dates for forecasting
future_dates = prophet_model.make_future_dataframe(periods=len(test), freq='M')

# Make predictions with Prophet
prophet_forecast = prophet_model.predict(future_dates)
prophet_forecast = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(test))

# Calculate Prophet metrics
prophet_rmse = sqrt(mean_squared_error(test, prophet_forecast['yhat']))
prophet_mae = mean_absolute_error(test, prophet_forecast['yhat'])
print(f"\nProphet RMSE: {prophet_rmse:.2f}")
print(f"Prophet MAE: {prophet_mae:.2f}")

# Plot the forecasts
plt.figure(figsize=(16, 8))
plt.plot(data.index, data['Passengers'], label='Actual', color='blue')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
plt.plot(test.index, prophet_forecast['yhat'].values, label='Prophet Forecast', color='green', linestyle='-.')
plt.fill_between(test.index, 
                prophet_forecast['yhat_lower'], 
                prophet_forecast['yhat_upper'], 
                color='green', alpha=0.2, label='Prophet Confidence Interval')
plt.title('Time Series Forecasting: ARIMA vs Prophet')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()

# Compare model performance
print("\nModel Comparison:")
comparison = pd.DataFrame({
    'Model': ['ARIMA', 'Prophet'],
    'RMSE': [arima_rmse, prophet_rmse],
    'MAE': [arima_mae, prophet_mae]
})
print(comparison)
