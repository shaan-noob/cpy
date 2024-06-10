import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Fetch historical stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))['Close']

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

time_step = 60
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Prepare training and testing data
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=1, validation_data=(X_test, y_test))

# Make predictions
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Plot actual vs. predicted stock prices
plt.figure(figsize=(16,8))
plt.plot(data.index, data.values, label='Actual Stock Price')
plt.plot(data.index[time_step:train_size], train_predict, label='Training Predictions')
plt.plot(data.index[train_size+time_step:], test_predict, label='Testing Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot training and validation loss and MAE
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(history.history['loss'], label='Training Loss', color='tab:blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('MAE')
ax2.plot(history.history['mae'], label='Training MAE', color='tab:green')
ax2.plot(history.history['val_mae'], label='Validation MAE', color='tab:red')
ax2.tick_params(axis='y')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Model Loss and MAE')
plt.show()

# Function to predict the next day's stock price
def predict_next_day(model, data, time_step):
    last_data = scaler.transform(data[-time_step:].values.reshape(-1, 1))
    next_day_prediction = model.predict(last_data.reshape(1, time_step, 1))
    return scaler.inverse_transform(next_day_prediction)[0, 0]

# Predict and print the next day's stock price
next_day_price = predict_next_day(model, data, time_step)
print(f'Next day predicted stock price: {next_day_price}')
