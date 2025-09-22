import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

extract_path = "/kaggle/input/animals10/raw-img"

img_size = (227, 227)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())

model = Sequential([
    Input(shape=(227, 227, 3)),
    Conv2D(96, (11, 11), strides=4, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),

    Conv2D(256, (5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),

    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((3, 3), strides=2),

    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

loss, accuracy = model.evaluate(val_generator)
print(f"Точність на валідації: {accuracy:.4f}")
print(f"Втрати (Loss): {loss:.4f}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
losses = history.history['loss']
val_losses = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точність на тренуванні')
plt.plot(epochs_range, val_acc, label='Точність на валідації')
plt.legend()
plt.title('Графік точності')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, losses, label='Втрати на тренуванні')
plt.plot(epochs_range, val_losses, label='Втрати на валідації')
plt.legend()
plt.title('Графік втрат')

plt.show()

random_class = random.choice(class_names)
random_image_path = random.choice(os.listdir(f"{extract_path}/{random_class}"))
img_path = f"{extract_path}/{random_class}/{random_image_path}"

img = image.load_img(img_path, target_size=(227, 227))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

plt.imshow(img)
plt.axis('off')
plt.title(f"Очікуваний: {random_class}\nПередбачений: {predicted_class}")
plt.show()

num_images = 1024
batch_size = 128
output_csv = "/kaggle/working/classification_results.csv"

all_images = []
for class_name in os.listdir(extract_path):
    class_dir = os.path.join(extract_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            all_images.append((img_path, class_name))

selected_images = random.sample(all_images, num_images)

def load_batch(image_data):
    images, paths, true_classes = [], [], []
    for img_path, true_class in image_data:
        img = image.load_img(img_path, target_size=(227, 227))
        img_array = image.img_to_array(img) / 255.0
        images.append(img_array)
        paths.append(img_path)
        true_classes.append(true_class)
    return np.array(images), paths, true_classes

results = []
for i in range(0, num_images, batch_size):
    batch_data = selected_images[i:i+batch_size]
    batch_images, batch_paths, batch_true_classes = load_batch(batch_data)
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    for j in range(len(batch_paths)):
        results.append([batch_paths[j], batch_true_classes[j], class_names[predicted_classes[j]]])

df = pd.DataFrame(results, columns=["Шлях до файлу", "Справжній клас", "Розпізнаний клас"])
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"Результати збережені у {output_csv}")
