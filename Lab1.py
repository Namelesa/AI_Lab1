from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
x_train = train.iloc[:, 1:].values.astype("float32") / 255.0
y_train = train.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test_cat),
    verbose=1
)
plt.plot(history.history["accuracy"], label="Точність на train")
plt.plot(history.history["val_accuracy"], label="Точність на test")
plt.xlabel("Епоха")
plt.ylabel("Точність")
plt.legend()
plt.show()

loss, accuracy = model.evaluate(x_test, y_test_cat)
print(f"Точність на тестових даних: {accuracy * 100:.2f}%")
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Передбачені мітки")
plt.ylabel("Справжні мітки")
plt.title("Матриця помилок")
plt.show()
def show_predictions(images, labels, preds, n=5):
    idxs = random.sample(range(len(images)), n)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(idxs):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Передбачено: {preds[idx]}\nПравильно: {labels[idx]}")
        plt.axis("off")
    plt.show()
show_predictions(x_test, y_test, y_pred_classes, n=5)