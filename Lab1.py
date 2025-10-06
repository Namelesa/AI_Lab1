import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import time

pos_dir = '/kaggle/input/apple-data/apple_images'
neg_dir = '/kaggle/input/cars-for-lab/BMW'
img_size = (299, 299)
batch_size = 8
epochs_stage1 = 10
epochs_stage2 = 20
num_val_files = 50
num_neg_train_files = 300
model_weights_download = True
video_path = '/kaggle/input/test-apple-video/Meet iPhone 16e _ Apple.mp4'
prediction_threshold = 0.5

def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        img.load()
        return True
    except Exception:
        return False

pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
             if is_valid_image(os.path.join(pos_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
             if is_valid_image(os.path.join(neg_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

print(f"Знайдено позитивних зображень (Apple): {len(pos_files)}")
print(f"Знайдено негативних зображень (BMW): {len(neg_files)}")

np.random.shuffle(pos_files)
np.random.shuffle(neg_files)

neg_files = neg_files[:num_neg_train_files]

val_pos = pos_files[:num_val_files]
val_neg = neg_files[:num_val_files]
train_pos = pos_files[num_val_files:]
train_neg = neg_files[num_val_files:]

val_files = val_pos + val_neg
val_labels = np.array([1] * len(val_pos) + [0] * len(val_neg))

train_files = train_pos + train_neg
train_labels = np.array([1] * len(train_pos) + [0] * len(train_neg))

print(f"\nТренувальний набір: {len(train_files)} (Apple: {len(train_pos)}, BMW: {len(train_neg)})")
print(f"Валідаційний набір: {len(val_files)} (Apple: {len(val_pos)}, BMW: {len(val_neg)})")

AUTOTUNE = tf.data.AUTOTUNE

def py_load_image(path_str):
    try:
        path = path_str.decode() if isinstance(path_str, (bytes, bytearray)) else str(path_str)
        img = Image.open(path).convert('RGB')
        img = img.resize((img_size[1], img_size[0]))
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape != (img_size[0], img_size[1], 3):
            arr = np.resize(arr, (img_size[0], img_size[1], 3))
        return arr
    except Exception:
        return np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)

def safe_load_image_tf(path):
    img = tf.numpy_function(py_load_image, [path], tf.float32)
    img.set_shape((img_size[0], img_size[1], 3))
    return img

def preprocess_map(path, label):
    img = safe_load_image_tf(path)
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_files), reshuffle_each_iteration=True)
train_ds = train_ds.map(preprocess_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
val_ds = val_ds.map(preprocess_map, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

def xception_block(x, depth_list, strides=(1,1), skip_type='conv', name_prefix=''):
    residual = x
    for i in range(3):
        x = SeparableConv2D(depth_list[i], (3,3), padding='same', use_bias=False,
                           name=f'{name_prefix}_sepconv{i+1}')(x)
        x = BatchNormalization(name=f'{name_prefix}_sepconv{i+1}_bn')(x)
        if i < 2:
            x = Activation('relu', name=f'{name_prefix}_sepconv{i+1}_act')(x)
    if strides != (1,1):
        x = MaxPooling2D((3,3), strides=strides, padding='same', name=f'{name_prefix}_pool')(x)
    if skip_type == 'conv':
        residual = Conv2D(depth_list[-1], (1,1), strides=strides, padding='same',
                         use_bias=False, name=f'{name_prefix}_skip_conv')(residual)
        residual = BatchNormalization(name=f'{name_prefix}_skip_bn')(residual)
    x = Add(name=f'{name_prefix}_add')([x, residual])
    x = Activation('relu', name=f'{name_prefix}_out_act')(x)
    return x

def build_xception(input_shape=(299,299,3), classes=1):
    img_input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), strides=(2,2), use_bias=False, padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3,3), use_bias=False, padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    x = xception_block(x, [128,128,128], strides=(2,2), skip_type='conv', name_prefix='block2')
    x = xception_block(x, [256,256,256], strides=(2,2), skip_type='conv', name_prefix='block3')
    x = xception_block(x, [728,728,728], strides=(2,2), skip_type='conv', name_prefix='block4')
    for i in range(8):
        residual = x
        prefix = f'block{5+i}'
        x = SeparableConv2D(728, (3,3), padding='same', use_bias=False, name=f'{prefix}_sepconv1')(x)
        x = BatchNormalization(name=f'{prefix}_sepconv1_bn')(x)
        x = Activation('relu', name=f'{prefix}_sepconv1_act')(x)
        x = SeparableConv2D(728, (3,3), padding='same', use_bias=False, name=f'{prefix}_sepconv2')(x)
        x = BatchNormalization(name=f'{prefix}_sepconv2_bn')(x)
        x = Activation('relu', name=f'{prefix}_sepconv2_act')(x)
        x = SeparableConv2D(728, (3,3), padding='same', use_bias=False, name=f'{prefix}_sepconv3')(x)
        x = BatchNormalization(name=f'{prefix}_sepconv3_bn')(x)
        x = Add(name=f'{prefix}_add')([x, residual])
        x = Activation('relu', name=f'{prefix}_out_act')(x)
    x = xception_block(x, [728,1024,1024], strides=(2,2), skip_type='conv', name_prefix='block13')
    x = SeparableConv2D(1536, (3,3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)
    x = SeparableConv2D(2048, (3,3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, activation='relu', name='custom_fc1')(x)
    x = Dropout(0.5, name='custom_dropout')(x)
    outputs = Dense(classes, activation='sigmoid' if classes==1 else 'softmax', name='custom_predictions')(x)
    return models.Model(img_input, outputs, name='xception')

print("\n" + "="*60)
print("СТВОРЕННЯ МОДЕЛІ XCEPTION")
print("="*60)

model = build_xception(input_shape=(img_size[0], img_size[1], 3), classes=1)

if model_weights_download:
    try:
        WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('xception_weights_notop.h5', WEIGHTS_URL, cache_subdir='models')
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print("✓ ImageNet ваги завантажені успішно (без top layer)")
    except Exception as e:
        print(f"✗ Не вдалося завантажити ваги: {e}")
        print("  Модель буде навчатися з випадковими вагами")

print("\n" + "="*60)
print("ЕТАП 1: НАВЧАННЯ КЛАСИФІКАТОРА (заморожені базові шари)")
print("="*60)

for layer in model.layers[:-3]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
total_params = sum([tf.size(w).numpy() for w in model.weights])
print(f"Тренувальних параметрів: {trainable_params:,} / {total_params:,}")

callbacks_stage1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history1 = model.fit(
    train_ds,
    epochs=epochs_stage1,
    validation_data=val_ds,
    callbacks=callbacks_stage1,
    verbose=1
)

print("\n" + "="*60)
print("ЕТАП 2: FINE-TUNING ВСІЄЇ МОДЕЛІ")
print("="*60)

for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Тренувальних параметрів: {trainable_params:,}")

callbacks_stage2 = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_xception_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

history2 = model.fit(
    train_ds,
    epochs=epochs_stage2,
    validation_data=val_ds,
    callbacks=callbacks_stage2,
    verbose=1
)

model.save('xception_custom_binary.h5')
print("\n✓ Модель збережена: xception_custom_binary.h5")

print("\n" + "="*60)
print("ГРАФІКИ НАВЧАННЯ")
print("="*60)

all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(all_loss, label='Train Loss', linewidth=2)
axes[0].plot(all_val_loss, label='Val Loss', linewidth=2)
axes[0].axvline(x=len(history1.history['loss']), color='red', linestyle='--', label='Fine-tuning start')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Функція втрат')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(all_acc, label='Train Accuracy', linewidth=2)
axes[1].plot(all_val_acc, label='Val Accuracy', linewidth=2)
axes[1].axvline(x=len(history1.history['accuracy']), color='red', linestyle='--', label='Fine-tuning start')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Точність')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Фінальна train accuracy: {all_acc[-1]:.4f}")
print(f"Фінальна val accuracy: {all_val_acc[-1]:.4f}")

print("\n" + "="*60)
print("ОЦІНКА МОДЕЛІ НА ВАЛІДАЦІЙНОМУ НАБОРІ")
print("="*60)

y_true, y_pred, y_probs = [], [], []
for batch_imgs, batch_labels in val_ds:
    probs = model.predict(batch_imgs, verbose=0)
    preds = (probs.flatten() > prediction_threshold).astype(int)
    y_true.extend(batch_labels.numpy().astype(int).tolist())
    y_pred.extend(preds.tolist())
    y_probs.extend(probs.flatten().tolist())

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BMW', 'Apple'],
            yticklabels=['BMW', 'Apple'])
plt.xlabel('Передбачено')
plt.ylabel('Фактично')
plt.title(f'Матриця помилок\nAcc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}, F1={f1:.2%}')
plt.tight_layout()
plt.show()

def process_video(video_file, model, img_size=(299,299), batch_size=32, threshold=0.5, fps_override=None):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError("Не вдалося відкрити відео: " + str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) if fps_override is None else fps_override
    if fps == 0:
        fps = 30
    periods, current_active, current_start = [], False, None
    frames, frame_times = [], []
    frame_idx = 0
    print(f"FPS відео: {fps:.2f}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, img_size)
        rgb = rgb.astype(np.float32) / 255.0
        frames.append(rgb)
        frame_times.append(t)
        if len(frames) >= batch_size:
            batch = np.array(frames)
            probs = model.predict(batch, verbose=0).flatten()
            for p, ft in zip(probs, frame_times):
                pred = int(p > threshold)
                if pred and not current_active:
                    current_active, current_start = True, ft
                elif (not pred) and current_active:
                    periods.append((current_start, ft))
                    current_active, current_start = False, None
            frames, frame_times = [], []
    if len(frames) > 0:
        batch = np.array(frames)
        probs = model.predict(batch, verbose=0).flatten()
        for p, ft in zip(probs, frame_times):
            pred = int(p > threshold)
            if pred and not current_active:
                current_active, current_start = True, ft
            elif (not pred) and current_active:
                periods.append((current_start, ft))
                current_active, current_start = False, None
    total_time = frame_idx / fps
    if current_active and current_start is not None:
        periods.append((current_start, total_time))
    cap.release()
    return periods

if os.path.exists(video_path):
    print("\n" + "="*60)
    print("ОБРОБКА ВІДЕО")
    print("="*60)
    t0 = time.time()
    periods = process_video(video_path, model, img_size=img_size, batch_size=32, threshold=prediction_threshold)
    elapsed = time.time() - t0
    print(f"\nЗнайдено періодів появи об'єкту: {len(periods)}")
    print("\nПеріоди (секунди):")
    for i, (s, e) in enumerate(periods, 1):
        duration = e - s
        print(f"  {i}. {s:.2f} - {e:.2f} (тривалість: {duration:.2f}s)")
    print(f"\nОброблено відео за {elapsed:.1f} сек")
else:
    print(f"\n✗ Файл відео не знайдено за шляхом: {video_path}")
    print("  Пропускаю етап тестування відео.")

print("\n" + "="*60)
print("ГОТОВО!")
print("="*60)
