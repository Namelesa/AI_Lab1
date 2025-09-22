import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from collections import Counter
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
DATASET_DIR = "dataset"
MODEL_NAME = "inception_transfer.h5"
CLASS_INDICES_PATH = "class_indices.json"
METADATA_PATH = "model_metadata.json"
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=0.2,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    for class_name, class_index in train_generator.class_indices.items():
        print(f"   {class_name} -> індекс {class_index}")
    return train_generator, val_generator
def create_callbacks():
    early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    checkpoint = callbacks.ModelCheckpoint(MODEL_NAME, monitor='val_loss', save_best_only=True)
    return [early, reduce, checkpoint]
def build_inception_v3_transfer(input_shape=(299, 299, 3), num_classes=1, freeze_base=True):
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    if freeze_base:
        base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name="InceptionV3_transfer")
    return model, base_model
def find_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    return float(thresholds[ix])
def train_model():
    train_generator, val_generator = create_data_generators()
    class_indices = train_generator.class_indices
    class_names = [""] * len(class_indices)
    for class_name, index in class_indices.items():
        class_names[index] = class_name
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "class_indices": class_indices}, f, ensure_ascii=False, indent=2)
    model, base_model = build_inception_v3_transfer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), freeze_base=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    callbacks_list = create_callbacks()
    counter = Counter(train_generator.classes)
    majority = max(counter.values())
    class_weight = {cls: float(majority / count) for cls, count in counter.items()}
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    val_generator.reset()
    y_prob = model.predict(val_generator, verbose=1).ravel()
    y_true = val_generator.classes
    best_thresh = find_best_threshold(y_true, y_prob)
    model.save(MODEL_NAME)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thresh}, f, ensure_ascii=False, indent=2)
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    ft_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max(5, EPOCHS // 2),
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    model.save(MODEL_NAME)
    return model, val_generator, class_names
def evaluate_model(model, val_generator, class_names):
    val_generator.reset()
    y_prob = model.predict(val_generator, verbose=1)
    y_prob = y_prob.ravel()
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = float(meta.get("threshold", 0.5))
    else:
        threshold = 0.5
    y_pred = (y_prob > threshold).astype(int)
    y_true = val_generator.classes
    y_pred_names = [class_names[pred] for pred in y_pred]
    y_true_names = [class_names[true] for true in y_true]
    acc = accuracy_score(y_true_names, y_pred_names)
    prec = precision_score(y_true_names, y_pred_names, labels=class_names, average="weighted", zero_division=0)
    rec = recall_score(y_true_names, y_pred_names, labels=class_names, average="weighted", zero_division=0)
    f1 = f1_score(y_true_names, y_pred_names, labels=class_names, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_names, y_pred_names, labels=class_names)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Матриця невідповідностей')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('Істинні')
    plt.xlabel('Прогнозовані')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "threshold": threshold}
def predict_image(model, image_path, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr, verbose=0).ravel()[0]
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = float(meta.get("threshold", 0.5))
    else:
        threshold = 0.5
    if prob > threshold:
        predicted = class_names[1]
        confidence = prob * 100
    else:
        predicted = class_names[0]
        confidence = (1 - prob) * 100
    return predicted, confidence
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print("❌ Папка з даними не знайдена:", DATASET_DIR)
    else:
        model, val_gen, class_names = train_model()
        results = evaluate_model(model, val_gen, class_names)
        test_dir = "dataset"
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(test_dir, fname)
                    predict_image(model, path, class_names)
        else:
            print("ℹ️ Папка test_images не знайдена. Додайте туди зображення для перевірки.")