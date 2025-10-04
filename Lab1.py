import os
import tensorflow as tf
from tensorflow.keras import layers, Model

TRAIN_ROOT = "/kaggle/input/cars-for-lab"
VAL_DIRS = {
    "BMW": "/kaggle/input/cars-for-lab/BMW",
    "Mercedes": "/kaggle/input/cars-for-lab/Mercedes"
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

def list_images(d):
    good_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
    files = [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith(good_ext) and os.path.isfile(os.path.join(d, f))
    ]
    valid_files = []
    for f in files:
        try:
            with open(f, 'rb') as img_file:
                header = img_file.read(12)
                if (header[:2] == b'\xff\xd8' or
                    header[:8] == b'\x89PNG\r\n\x1a\n' or
                    header[:6] in (b'GIF87a', b'GIF89a') or
                    header[:2] == b'BM'):
                    valid_files.append(f)
        except:
            pass
    return valid_files

all_classes = [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
classes = sorted(all_classes)
class_index = {c: i for i, c in enumerate(classes)}

print(f"Знайдені класи: {classes}\n")

train_files = []
train_labels = []

for cls in classes:
    cls_dir = os.path.join(TRAIN_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    idx = class_index[cls]
    images = list_images(cls_dir)
    for p in images:
        train_files.append(p)
        train_labels.append(idx)
    print(f"Клас {cls}: додано {len(images)} файлів для навчання")

val_files = []
val_labels = []

for cls, cls_dir in VAL_DIRS.items():
    if not os.path.isdir(cls_dir):
        print(f"УВАГА: Директорія {cls_dir} не знайдена!")
        continue
    idx = class_index[cls]
    images = list_images(cls_dir)
    for p in images:
        val_files.append(p)
        val_labels.append(idx)
    print(f"Клас {cls}: додано {len(images)} файлів для валідації")

print(f"\n✅ Всього для навчання: {len(train_files)} файлів")
print(f"✅ Всього для валідації: {len(val_files)} файлів")

def make_dataset(file_paths, labels, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
        img = img / 255.0
        return img, label
    ds = ds.map(lambda p, l: _load(p, l), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_files, train_labels, shuffle=True, augment=True)
val_ds = make_dataset(val_files, val_labels, shuffle=False, augment=False)

input_tensor = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", input_tensor=input_tensor
)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(len(classes), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

print("\n🚀 Починаємо навчання...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n🔧 Fine-tuning: розморожуємо останні шари...\n")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model.save("/kaggle/working/inception_custom.h5")
print("\n✅ Модель збережена!")

print(f"\nФінальна точність на навчанні: {history_fine.history['accuracy'][-1]:.4f}")
print(f"Фінальна точність на валідації: {history_fine.history['val_accuracy'][-1]:.4f}")
