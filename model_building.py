import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
import pickle

# ------------------ 1. Load & Clean Dataset ------------------
df = pd.read_csv("Dataset.csv")

def clean_code(text):
    return text.strip().replace('\n', ' ').replace('\t', ' ')

df['code'] = df['code'].apply(clean_code)

# ------------------ 2. Tokenize Code ------------------
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df['code'])

# Save tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✅ Tokenizer saved as 'tokenizer.pkl'.")

# Convert to sequences
sequences = tokenizer.texts_to_sequences(df['code'])
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

labels = np.array(df['stderr'])  # 0: Bug-Free, 1: Buggy

# ------------------ 3. Handle Class Imbalance ------------------
ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = ros.fit_resample(padded_sequences, labels)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

# ------------------ 4. Build Model ------------------
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=256, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# ------------------ 5. Callbacks ------------------
callbacks = [
    ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
]

# ------------------ 6. Train Model ------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ------------------ 7. Evaluate ------------------
loss, accuracy = model.evaluate(X_val, y_val)
print(f"✅ Validation Accuracy: {accuracy:.4f}")

# ------------------ 8. Save Final Model ------------------
model.save("Trained_model.h5")
print("✅ Model saved as 'Trained_model.h5'.")
