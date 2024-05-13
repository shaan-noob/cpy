import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def activation_functions(x):
    return {
        'tanh': np.tanh(x),
        'sigmoid': 1 / (1 + np.exp(-x)),
        'relu': np.maximum(0, x),
        'softmax': np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    }

# Generate input data for plotting
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(12, 6))
for i, (activation_name, activation_function) in enumerate(activation_functions(x).items(), start=1):
    plt.subplot(2, 2, i)
    plt.plot(x, activation_function)
    plt.title(f'{activation_name.capitalize()} Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate parameters
input_dim, hidden_units, output_units = 784, [128, 64, 32], 10
input_layer_params = (input_dim + 1) * hidden_units[0]  # Add 1 for the bias term
hidden_layer_params = sum((hidden_units[i] + 1) * hidden_units[i+1] for i in range(len(hidden_units) - 1))
output_layer_params = (hidden_units[-1] + 1) * output_units  # Add 1 for the bias term
total_params = input_layer_params + hidden_layer_params + output_layer_params
print("Total number of parameters:", total_params)
