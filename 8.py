import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Plot accuracy and loss
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(history.history['loss'], label='Training Loss', color='tab:blue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
ax2 = ax1.twinx()
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='tab:green')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:red')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Model Loss and Accuracy')
plt.show()

# Display predictions with images
predictions = np.argmax(model.predict(X_test), axis=1)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predictions[i]}")
    plt.axis('off')
plt.show()
