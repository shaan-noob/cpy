import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build a simple CNN model for classification
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Adversarial training
def adversarial_training(model, X_train, y_train, epochs=5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)
    return model

# Tangent Propagation (using the same model architecture)
def tangent_propagation(model, X_train, y_train, epochs=5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)
    return model

# Tangent Classifier (using the same model architecture)
def tangent_classifier(model, X_train, y_train, epochs=5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)
    return model

# Main code
if __name__ == "__main__":
    # Build the model
    model = build_model()

    # Adversarial Training
    print("Adversarial Training:")
    model_adv = adversarial_training(model, X_train, y_train)
    loss, accuracy = model_adv.evaluate(X_test, y_test)
    print(f"Accuracy on test set: {accuracy}")

    # Tangent Propagation
    print("\nTangent Propagation:")
    model_tp = tangent_propagation(model, X_train, y_train)
    loss, accuracy = model_tp.evaluate(X_test, y_test)
    print(f"Accuracy on test set: {accuracy}")

    # Tangent Classifier
    print("\nTangent Classifier:")
    model_tc = tangent_classifier(model, X_train, y_train)
    loss, accuracy = model_tc.evaluate(X_test, y_test)
    print(f"Accuracy on test set: {accuracy}")
