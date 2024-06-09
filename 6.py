import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Generate and save random stock price data
num_days, price_start = 1000, 100
dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(num_days)]
prices = price_start + np.cumsum(np.random.normal(0, 1, num_days))
pd.DataFrame({'Date': dates, 'Close': prices}).to_csv('random_stock_prices.csv', index=False)

# Load and preprocess data
data = pd.read_csv('random_stock_prices.csv', index_col='Date', parse_dates=['Date'])
data = data[['Close']]
scaled_data = MinMaxScaler().fit_transform(data)
train_len = int(len(scaled_data) * 0.8)

# Prepare training and testing data
def create_dataset(data, start, end):
    x, y = [], []
    for i in range(start, end):
        x.append(data[i-60:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(scaled_data, 60, train_len)
x_test, y_test = create_dataset(scaled_data, train_len, len(scaled_data))
x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

# Build and train the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Make predictions
predictions = MinMaxScaler().fit(data).inverse_transform(model.predict(x_test))

# Plot the results
plt.figure(figsize=(16, 8))
plt.plot(data.index[:train_len], data['Close'][:train_len], label='Train')
plt.plot(data.index[train_len:], data['Close'][train_len:], label='Val')
plt.plot(data.index[train_len:], predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()
