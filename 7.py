import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Generate synthetic stock price data
def generate_synthetic_data(num_samples):
    np.random.seed(42)
    return np.sin(np.arange(num_samples) * 0.01) + np.random.normal(0, 0.5, num_samples)

# Prepare the dataset
def prepare_data(data, time_steps):
    X = [data[i:(i + time_steps)] for i in range(len(data) - time_steps)]
    y = data[time_steps:]
    return np.array(X), np.array(y)

# Parameters
num_samples = 1000
time_steps = 10

# Generate and scale synthetic data
data = generate_synthetic_data(num_samples)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Prepare the dataset
X, y = prepare_data(data_scaled, time_steps)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train the GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(time_steps, 1)),
    GRU(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions
predicted_stock_price = scaler.inverse_transform(model.predict(X_test))

# Plot the results
plt.plot(scaler.inverse_transform(data_scaled[split + time_steps:]), color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
