import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras.src.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Step 1: Load the labeled dataset
data = pd.read_csv('training_data.csv')  # Assuming the dataset is in a CSV file

# Step 2: Split the dataset into input features (error messages) and target variable (bug/error types)
X = data['error_message']
y = data['bug_or_error_type']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Tokenization and sequence padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max([len(sequence) for sequence in X_train_sequences])
vocab_size = len(tokenizer.word_index) + 1

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post')

# Step 5: Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

y_train_categorical = to_categorical(y_train_encoded, num_classes)
y_test_categorical = to_categorical(y_test_encoded, num_classes)

# Step 6: Model training (RNN with LSTM)
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train_categorical, validation_data=(X_test_padded, y_test_categorical), epochs=10, batch_size=32)

# Step 7: Model evaluation
loss, accuracy = model.evaluate(X_test_padded, y_test_categorical, verbose=0)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

sequences = tokenizer.texts_to_sequences(["Assertion"])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
model.save("bug_error_modle.h5")

# Calling script

# Function to preprocess input text
def preprocess_text(text):
    with open('tokenizer.json', 'r') as f:
        tokenizer = tokenizer_from_json(f.read())
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=15, padding='post')
    return padded_sequences

# Function to make predictions
def predict_bug_or_error(text):
    model = load_model("bug_error_modle.h5")
    preprocessed_text = preprocess_text(text)
    predictions = model.predict(preprocessed_text)
    predicted_label = np.argmax(predictions[0])
    label_encoder.fit_transform(y)
    return label_encoder.classes_[predicted_label]

# Example usage
input_text = "ElementNotSelectableException: Element is not selectable  "
predicted_bug_or_error = predict_bug_or_error(input_text)
print(f"Predicted Bug or Error: {predicted_bug_or_error}")