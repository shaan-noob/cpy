import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and inspect data
data = pd.read_csv("/content/yahoo_stock (1).csv")
ts_data = data[['Date', 'Open']]
ts_data['Date'] = pd.to_datetime(ts_data['Date'], format="%Y-%m-%d")
ts_data.set_index('Date', inplace=True)

# Plot opening prices
plt.figure(figsize=(10, 6))
plt.plot(ts_data.index, ts_data['Open'])
plt.ylabel("Price")
plt.xlabel("Date")
plt.title("Opening Price of the Stocks")
plt.xticks(rotation=45)
plt.xlim(ts_data.index.min(), ts_data.index.max())
plt.show()

# Decompose and plot
decompose_result = seasonal_decompose(ts_data['Open'], model='multiplicative', period=363)
decompose_result.plot()
plt.show()
