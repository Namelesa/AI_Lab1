import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

train_path = "/kaggle/input/yelp-review-polarity/yelp_review_polarity_csv/train.csv"
test_path = "/kaggle/input/yelp-review-polarity/yelp_review_polarity_csv/test.csv"

train_df = pd.read_csv(train_path, header=None)
test_df = pd.read_csv(test_path, header=None)

train_df.columns = ["label", "text"]
test_df.columns = ["label", "text"]

print("Перші 3 рядки train_df:")
print(train_df.head(3))
print(f"\nУнікальні мітки: {train_df['label'].unique()}")
print(f"Розмір train: {train_df.shape}, test: {test_df.shape}")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

x_train = tokenizer.texts_to_sequences(train_df["text"])
x_test = tokenizer.texts_to_sequences(test_df["text"])

x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

y_train = np.array(train_df["label"]) - 1
y_test = np.array(test_df["label"]) - 1

print(f"\n✅ Дані готові: x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n" + "="*50)
print(model.summary())
print("="*50 + "\n")

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\n🔍 Оцінка моделі на тестових даних...")
y_pred = (model.predict(x_test) > 0.5).astype("int32")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("📊 РЕЗУЛЬТАТИ МОДЕЛІ:")
print("="*50)
print(f"✅ Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall:    {rec:.4f}")
print(f"✅ F1-score:  {f1:.4f}")
print("="*50 + "\n")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Негативний', 'Позитивний'],
            yticklabels=['Негативний', 'Позитивний'])
plt.title("Матриця помилок LSTM", fontsize=16, fontweight='bold')
plt.xlabel("Прогноз", fontsize=12)
plt.ylabel("Справжнє значення", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Точність моделі')
plt.xlabel('Епоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Втрати моделі')
plt.xlabel('Епоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("🧪 ТЕСТУВАННЯ НА НОВИХ ПРИКЛАДАХ:")
print("="*50)

samples = [
    "This restaurant was amazing! The food was delicious and the service was great.",
    "Terrible experience, I will never come back again.",
    "The movie was just okay, nothing special.",
    "Absolutely loved this place, 10/10!",
    "Worst food ever. Very disappointed."
]

seqs = tokenizer.texts_to_sequences(samples)
seqs = pad_sequences(seqs, maxlen=max_len, padding='post')

preds = model.predict(seqs, verbose=0)
for i, (txt, p) in enumerate(zip(samples, preds), 1):
    label = "Позитивний 😊" if p > 0.5 else "Негативний 😞"
    confidence = p[0] if p > 0.5 else 1 - p[0]
    print(f"\n{i}. 📝 {txt}")
    print(f"   → {label} (впевненість: {confidence:.2%})")

print("\n" + "="*50)
print("✅ Тестування завершено!")
print("="*50)
