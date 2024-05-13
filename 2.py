import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

# Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape((-1, 28 * 28)).astype('float32') / 255.0, X_test.reshape((-1, 28 * 28)).astype('float32') / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define a function to create and train the model
def train_model(optimizer):
    model = Sequential([Dense(128, input_shape=(784,), activation='relu'), Dense(64, activation='relu'), Dense(10, activation='softmax')])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Train the model using gradient descent and stochastic gradient descent
gd_history = train_model('sgd')
sgd_history = train_model(SGD(learning_rate=0.01, momentum=0.9))

# Plot the training and validation accuracy for both optimizers
plt.plot(gd_history.history['accuracy'], label='Gradient Descent (Train)')
plt.plot(gd_history.history['val_accuracy'], label='Gradient Descent (Validation)')
plt.plot(sgd_history.history['accuracy'], label='Stochastic Gradient Descent (Train)')
plt.plot(sgd_history.history['val_accuracy'], label='Stochastic Gradient Descent (Validation)')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
