import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset and normalize pixel values
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define input shape and shared input layer
input_layer = Input(shape=X_train[0].shape)
flatten_layer = Flatten()(input_layer)

# Shared hidden layer
shared_hidden = Dense(64, activation='relu')(flatten_layer)

# Task-specific output layers
coarse_output = Dense(10, activation='softmax', name='coarse_output')(shared_hidden)
fine_output = Dense(10, activation='softmax', name='fine_output')(shared_hidden)

# Define the model
model = Model(inputs=input_layer, outputs=[coarse_output, fine_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'coarse_output': 'sparse_categorical_crossentropy', 'fine_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
history = model.fit(X_train, {'coarse_output': y_train, 'fine_output': y_train},
                    validation_split=0.2, epochs=100, callbacks=[early_stopping])

# Plot training and validation loss for coarse classification
for output_type in ['coarse_output', 'fine_output']:
    train_loss = history.history[f'{output_type}_loss']
    val_loss = history.history[f'val_{output_type}_loss']
    plt.plot(range(1, len(train_loss) + 1), train_loss, label=f'{output_type.capitalize()} Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label=f'{output_type.capitalize()} Validation Loss')

# Indicate early stopping point on the plot
if early_stopping.stopped_epoch != 0:
    plt.axvline(x=early_stopping.stopped_epoch + 1, color='r', linestyle='--', label='Early Stopping')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
