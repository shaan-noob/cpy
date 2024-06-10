import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import datetime

# Load and prepare the data
ticker = 'AAPL'  # Example ticker
data = yf.download(ticker, start='2010-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))['Close']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Create datasets
time_step = 60
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Build and train the GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(time_step, 1)),
    GRU(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=5, batch_size=1, validation_data=(X_test, y_test))

# Make predictions
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Plot stock prices and predictions
plt.figure(figsize=(16, 8))
plt.plot(data.index, data.values, label='Actual Stock Price')
plt.plot(data.index[time_step:train_size], train_predict, label='Training Predictions')
plt.plot(data.index[train_size + time_step:], test_predict, label='Testing Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot loss and MAE in one graph
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(history.history['loss'], label='Training Loss', color='tab:blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
ax2 = ax1.twinx()
ax2.set_ylabel('MAE')
ax2.plot(history.history['mae'], label='Training MAE', color='tab:green')
ax2.plot(history.history['val_mae'], label='Validation MAE', color='tab:red')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Model Loss and MAE')
plt.show()

# Real-time prediction
def predict_next_day(model, data, time_step):
    last_data = scaler.transform(data[-time_step:].values.reshape(-1, 1))
    next_day_prediction = model.predict(last_data.reshape(1, time_step, 1))
    return scaler.inverse_transform(next_day_prediction)[0, 0]

next_day_price = predict_next_day(model, data, time_step)
print(f'Next day predicted stock price: {next_day_price}')
