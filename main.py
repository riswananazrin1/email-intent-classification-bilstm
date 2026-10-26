import pandas as pd
data = pd.read_csv("dataset/cleaned_email_intent.csv")
print("Dataset loaded  successfully!")
print("Number of rows:",len(data))
print("Number of columns:",len(data.columns))
print("\nColumn names:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())
print("\nIntent distribution:")
print(data["intent"].value_counts())
print("\nMissing values:")
print(data.isnull().sum())
print("\nNumber of duplicate emails:")
print(data["email"].duplicated().sum())
duplicate_intents = (data.groupby("email")["intent"].nunique())
print("\nEmails with multiple intent labels:")
print((duplicate_intents > 1).sum())
data = data.drop_duplicates(subset="email")
print("\nRows after removing duplicates:")
print(len(data))
data.to_csv("Dataset/cleaned_email_intent.csv",index=False)
print("\nCleaned dataset saved successfully!")
x= data["email"]
y = data["intent"]
print("\nInput data (X):")
print(x.head())
print("\nTarget data (y):")
print(y.head())
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nIntent labels:")
print(label_encoder.classes_)
print("\nEncoded intent labels:")
print(y_encoded[:10])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    x,
    y_encoded,
    test_size = 0.2,
    random_state=42,
    stratify =y_encoded

)
print("\nTraining samples:",len(X_train))
print("Testing samples:",len(X_test))
from tensorflow.keras.preprocessing.text import Tokenizer


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
print("\nVocabulary size:", len(tokenizer.word_index))
print("\nFirst training email:")
print(X_train.iloc[0])
print("\nTokenized version:")
print(X_train_sequences[0])
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_length = max(len(sequence) for sequence in X_train_sequences)


X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding="post"
)


X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding="post"
)

print("\nMaximum sequence length:", max_length)
print("Training data shape:", X_train_padded.shape)
print("Testing data shape:", X_test_padded.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense

vocab_size = len(tokenizer.word_index) + 1


num_classes = len(label_encoder.classes_)


model = Sequential([
    Input(shape=(max_length,)),
    Embedding(input_dim=vocab_size, output_dim=64),
    Bidirectional(LSTM(64)),
    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nBiLSTM model created successfully!")

model.summary()
history = model.fit(
    X_train_padded,
    y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2

)
test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    y_test,
    verbose = 1
)
print("\nTest Loss:", test_loss)
print("Test Accuracy:",test_accuracy)
from sklearn.metrics import classification_report
y_pred_prob = model.predict(X_test_padded)
y_pred = y_pred_prob.argmax(axis = 1)
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
target_names=label_encoder.classes_
    )
    )
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)


for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error
import numpy as np


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMSE:", mse)
print("RMSE:", rmse)
model.save("email_intent_bilstm.keras")
print("\nModel saved successfully!")
new_email = ["I am unhappy with the poor service I received."]

new_sequence = tokenizer.texts_to_sequences(new_email)

new_padded = pad_sequences(
    new_sequence,
    maxlen=max_length,
    padding="post"
)

prediction = model.predict(new_padded)

predicted_class = prediction.argmax(axis=1)[0]

predicted_intent = label_encoder.inverse_transform(
    [predicted_class]
)[0]

print("\nNew Email:", new_email[0])
print("Predicted Intent:", predicted_intent)
           