---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# **Machine Learning for Time Series Data**

Time series data is an ordered sequence of data points collected or recorded at specific time intervals. Examples include stock prices, weather data, sensor readings, sales data, and website traffic. Unlike traditional datasets, time series data has **temporal dependencies** that must be captured to make accurate predictions.

In this section, we'll cover:

1. **Types of Time Series Problems**
2. **Key Challenges in Time Series Analysis**
3. **Popular Machine Learning Algorithms for Time Series**
4. **Deep Learning Approaches for Time Series**
5. **Project: Forecasting Stock Prices**
6. **Evaluation Metrics for Time Series**

---

## **1. Types of Time Series Problems**

### **1.1. Time Series Forecasting**:
The goal is to predict future values based on historical data. For example, predicting the temperature for the next 7 days, forecasting stock prices, or estimating future product demand.

### **1.2. Time Series Classification**:
Given a sequence of data, the goal is to classify the entire series or detect patterns within the series. For example, classifying an ECG signal as normal or abnormal or detecting fraudulent transactions.

### **1.3. Anomaly Detection**:
Identifying outliers or abnormal patterns in time series data. This is important in applications like network security (detecting unusual traffic), sensor monitoring (identifying faulty equipment), or financial systems (detecting fraudulent activities).

### **1.4. Time Series Regression**:
Predicting a continuous output where the input includes one or more time series variables. This is similar to regular regression but takes into account the time dependencies.

---

## **2. Key Challenges in Time Series Analysis**

- **Temporal Dependency**: Time series data points are not independent of each other. Each point may be influenced by previous time steps, and capturing this dependency is crucial.
- **Trend and Seasonality**: Time series data often exhibit trends (upward or downward movements) and seasonal patterns (regular fluctuations that repeat over time).
- **Stationarity**: Many machine learning models assume that the data is stationary (its statistical properties do not change over time), which may not be the case for many real-world time series.
- **Missing Data**: Time series data often contains missing values, which need to be handled carefully, as they can affect the model's predictions.
- **Multivariate Time Series**: In some cases, multiple time series variables (e.g., temperature, humidity, wind speed) are correlated and need to be considered together.

---

## **3. Popular Machine Learning Algorithms for Time Series**

Several traditional machine learning algorithms can be adapted for time series forecasting, classification, and regression. These include:

### **3.1. ARIMA (AutoRegressive Integrated Moving Average)**
- **ARIMA** is a widely used statistical method for time series forecasting. It combines three components: autoregression (AR), differencing to remove non-stationarity (I), and moving average (MA).
  
- **ARIMA** is suited for univariate time series data and is effective for short-term forecasting when the underlying data is stationary.

**Example**:
```{code-cell} python
from statsmodels.tsa.arima.model import ARIMA

# Example univariate time series data (e.g., stock prices)
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119]

# Fit an ARIMA model
model = ARIMA(data, order=(1, 1, 1))  # (p, d, q) parameters
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=5)
print(forecast)
```

#### **3.2. Exponential Smoothing (ETS)**
- **Exponential Smoothing** methods (e.g., Holt-Winters) are used for forecasting time series data with trends and seasonality. The model captures the level, trend, and seasonality of the data.
  
- It is particularly useful for making short-term forecasts.

**Example**:
```{code-cell} python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Example time series data with trend and seasonality
data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119]

# Fit an Exponential Smoothing model (Additive Trend and Seasonality)
model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=4)
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=5)
print(forecast)
```

#### **3.3. Random Forest for Time Series**
- **Random Forest** can be adapted to time series forecasting by using lagged values of the time series as features. For example, to predict \(y_t\), you might use \(y_{t-1}\), \(y_{t-2}\), etc., as input features.

- **Random Forest** can capture non-linear relationships between lagged variables, making it more flexible than linear models like ARIMA.

**Example**:
```{code-cell} python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Generate time-lagged features
data = pd.DataFrame({'value': [112, 118, 132, 129, 121, 135, 148, 148, 136, 119]})
data['lag1'] = data['value'].shift(1)
data['lag2'] = data['value'].shift(2)
data.dropna(inplace=True)

# Train a Random Forest model
X = data[['lag1', 'lag2']]
y = data['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print(y_pred)
```

#### **3.4. XGBoost for Time Series**
- **XGBoost** is a gradient-boosting algorithm that can be adapted for time series forecasting. Similar to Random Forest, you create lagged features to capture the temporal dependencies in the data.

- XGBoost is powerful for multivariate time series or datasets with complex relationships between variables.

**Example**:
```{code-cell} python
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

# Load and preprocess data
# Assume data is a pandas DataFrame with a 'value' column and lagged features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f'RMSE: {rmse}')
```

---

### **4. Deep Learning Approaches for Time Series**

#### **4.1. Recurrent Neural Networks (RNNs)**
- **RNNs** are designed to handle sequential data by passing the hidden state from one time step to the next. This makes them well-suited for time series forecasting and classification tasks where temporal dependencies are key.

#### **4.2. Long Short-Term Memory (LSTM)**
- **LSTMs** are a type of RNN designed to handle longer sequences by mitigating the vanishing gradient problem. They are widely used for time series forecasting when long-term dependencies are crucial.

**Example: LSTM for Time Series Forecasting**

```{code-cell} python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import Input

# Generate synthetic time series data
data = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119])
data = data.reshape((len(data), 1, 1))  # Reshape for LSTM input

# Build LSTM model
model = Sequential()
model.add(Input((1, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(data[:-1], data[1:], epochs=200, verbose=0)

# Predict the next value
x_input = np.array([119]).reshape((1, 1, 1))
y_pred = model.predict(x_input)
print(y_pred)
```

#### **4.3. Convolutional Neural Networks (CNNs) for Time Series**
- **CNNs** can be applied to time series by treating the data as 1D signals. By applying convolutional filters, CNNs can detect short-term patterns and features in time series data.

#### **4.4. Transformer Models for Time Series**
- **Transformers**, originally developed for NLP tasks, can also be used for time series forecasting. They rely on attention mechanisms to capture dependencies between different time steps, making them effective for multivariate time series with complex temporal relationships.

---

### **5. Project: Forecasting Stock Prices Using LSTM**

#### **Goal**:
Predict future stock prices using LSTM based on historical prices.

#### **Dataset**:
Use the **Yahoo Finance API** to fetch historical stock prices for a company (e.g., Apple). You can use the `yfinance` library to easily access stock data.

#### **Steps**:
1. **Data Preprocessing**:
   - Fetch historical stock prices (e.g., "Close" prices) using `yfinance`.
   - Normalize the data

 using MinMax scaling.
   - Create sequences of historical prices to be used as input for the LSTM model.

2. **LSTM Model**:
   - Build an LSTM model using **Keras** or **PyTorch**.
   - Train the model to predict the next stock price based on the previous prices in the sequence.

3. **Evaluation**:
   - Evaluate the model using **Root Mean Squared Error (RMSE)** or **Mean Absolute Error (MAE)**.

#### **Code Example: Stock Price Forecasting Using LSTM**

```{code-cell} python
:tags: [scroll-output]

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import Input

# Fetch historical stock prices (e.g., Apple)
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare the data (create sequences of 60 days)
X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

# Build the LSTM model
model = Sequential()
model.add(Input((X.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Predict future stock prices
y_pred = model.predict(X)
y_pred_scaled = scaler.inverse_transform(y_pred)

print(y_pred_scaled.reshape(-1)[:10])
```

---

### **6. Evaluation Metrics for Time Series**

- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: A more sensitive metric to large errors than MAE.
- **Mean Absolute Percentage Error (MAPE)**: A percentage-based error metric useful for comparing across datasets with different scales.
- **R-squared**: Measures the proportion of variance explained by the model.

---

### **Conclusion**

Time series analysis requires specialized models and methods due to its temporal structure. Traditional techniques like **ARIMA** and **Exponential Smoothing** are still effective for short-term univariate forecasting, while **machine learning models** like **Random Forest** and **XGBoost** are useful for more complex multivariate series. **Deep learning models** such as **LSTMs**, **CNNs**, and **Transformers** have become popular for capturing long-term dependencies and patterns in time series data.
