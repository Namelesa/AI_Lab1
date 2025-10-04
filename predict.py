import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = "/kaggle/input/cars-for-lab"
MODEL_PATH = "/kaggle/working/inception_custom.h5"
IMG_SIZE = (224, 224)

classes = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
print(f"Класи: {classes}\n")

model = tf.keras.models.load_model(MODEL_PATH)


def load_and_predict(file_path, model, classes):
    try:
        img = image.load_img(file_path, target_size=IMG_SIZE)
        arr = image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        return True, classes[idx], conf
    except:
        return False, None, None


for cat in classes:
    path = os.path.join(BASE_DIR, cat)
    files = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(files) == 0:
        print(f"⚠️ У папці {cat} немає зображень")
        continue

    print(f"🔹 Клас: {cat}")
    random.shuffle(files)

    successful_predictions = 0
    attempted_files = 0
    max_attempts = min(len(files), 20)

    for fn in files:
        if successful_predictions >= 5:
            break
        if attempted_files >= max_attempts:
            break

        attempted_files += 1
        p = os.path.join(path, fn)
        success, predicted_class, conf = load_and_predict(p, model, classes)

        if success:
            successful_predictions += 1
            is_correct = "✅" if predicted_class == cat else "❌"
            print(f"   {is_correct} {fn} → {predicted_class} ({conf * 100:.2f}%)")
        else:
            print(f"   ⚠️ {fn} → ПОМИЛКА завантаження, пропущено")

    if successful_predictions == 0:
        print(f"   ❌ Не вдалося завантажити жодного зображення з папки {cat}")
    elif successful_predictions < 5:
        print(f"   ℹ️ Вдалося обробити лише {successful_predictions} зображень")

    print()

print("✅ Передбачення завершено!")
