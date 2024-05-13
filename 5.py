import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample data
texts = ["I love this movie!", "This movie is fantastic.", "I didn't like this film.",
         "Terrible acting, awful movie.", "Not bad, I enjoyed it."]
labels = np.array([1, 1, 0, 0, 1])  # 1 for positive, 0 for negative

max_words, max_len, embedding_dim = 1000, 50, 50

# Tokenization and padding
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
padded_sequences = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

# Model definition and training
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
    Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10, batch_size=32)

# Testing
test_texts = ["This movie is amazing!", "Terrible acting, awful movie."]
padded_test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len)
predictions = model.predict(padded_test_sequences)

# Output predictions
for text, sentiment in zip(test_texts, predictions):
    print(text, ": Positive" if sentiment > 0.5 else ": Negative")
