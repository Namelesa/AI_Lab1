import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

apple_dir = '/kaggle/input/apple-data/apple_images'
bmw_dir = '/kaggle/input/cars-for-lab/BMW'
img_size = (299, 299)
video_path = '/kaggle/input/test-apple-video/Meet iPhone 16e _ Apple.mp4'
prediction_threshold = 0.5
num_test_images = 10

print("=" * 60)
print("ЗАВАНТАЖЕННЯ МОДЕЛІ")
print("=" * 60)

try:
    model = tf.keras.models.load_model('xception_custom_binary.h5')
    print("✓ Модель успішно завантажена: xception_custom_binary.h5")
except Exception as e:
    print(f"✗ Помилка завантаження моделі: {e}")
    exit(1)

class_names = {1: 'Apple', 0: 'not Apple'}


def is_valid_image(filepath):
    try:
        img = Image.open(filepath)
        img.verify()
        img = Image.open(filepath)
        img.convert('RGB').load()
        return True
    except Exception:
        return False


def predict_images(model, paths, true_labels, show_images=True):
    correct = 0
    total = 0
    images_to_show = []
    predictions_detail = []

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ ЗОБРАЖЕНЬ")
    print("=" * 60)
    print(f"{'Статус':<6} {'Файл':<35} {'Передбачення':<12} {'Впевненість':<12} {'Істина':<12}")
    print("-" * 90)

    for img_path, true_label in zip(paths, true_labels):
        try:
            img = load_img(img_path, target_size=img_size)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = model.predict(x, verbose=0)[0][0]
            predicted_class = int(pred > prediction_threshold)
            predicted_label = class_names[predicted_class]
            confidence = pred if predicted_class == 1 else (1 - pred)

            is_correct = (predicted_class == true_label)
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            filename = os.path.basename(img_path)[:35]
            true_label_name = class_names[true_label]

            print(f"{status:<6} {filename:<35} {predicted_label:<12} {confidence:>10.2%}  {true_label_name:<12}")

            images_to_show.append((img, predicted_label, confidence, is_correct, true_label_name))
            predictions_detail.append({
                'path': img_path,
                'true_label': true_label,
                'pred_class': predicted_class,
                'pred_label': predicted_label,
                'confidence': confidence,
                'correct': is_correct
            })

        except Exception as e:
            print(f"✗ Помилка при обробці {os.path.basename(img_path)}: {e}")

    print("-" * 90)
    if total > 0:
        accuracy = correct / total
        print(f"ТОЧНІСТЬ: {correct}/{total} ({accuracy * 100:.1f}%)")
    else:
        accuracy = 0.0
        print("ТОЧНІСТЬ: Немає валідних зображень для обробки!")
    print("=" * 60)

    if show_images and images_to_show:
        n = len(images_to_show)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (img, pred_label, conf, correct, true_label) in enumerate(images_to_show):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            ax.imshow(img)
            ax.axis('off')

            color = 'green' if correct else 'red'
            title = f"{pred_label}\n{conf:.1%} {'✓' if correct else '✗'}\n(True: {true_label})"
            ax.set_title(title, color=color, fontsize=10, fontweight='bold')

        for i in range(len(images_to_show), rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return accuracy, predictions_detail


def process_video(video_file, model, img_size=(299, 299), batch_size=32, threshold=0.5):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"✗ Не вдалося відкрити відео: {video_file}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        print(f"⚠ FPS не визначено, використовую {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps:.2f}")
    print(f"Загальна кількість кадрів: {total_frames}")
    print(f"Тривалість відео: {total_frames / fps:.2f} сек")

    periods = []
    current_active = False
    current_start = None
    frames = []
    frame_times = []
    frame_idx = 0

    print("\nОбробка відео...")
    last_progress = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        frame_idx += 1

        progress = int((frame_idx / total_frames) * 100)
        if progress >= last_progress + 10:
            print(f"  Прогрес: {progress}% ({frame_idx}/{total_frames} кадрів)")
            last_progress = progress

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
                    current_active = True
                    current_start = ft
                elif (not pred) and current_active:
                    periods.append((current_start, ft))
                    current_active = False
                    current_start = None

            frames, frame_times = [], []

    if len(frames) > 0:
        batch = np.array(frames)
        probs = model.predict(batch, verbose=0).flatten()

        for p, ft in zip(probs, frame_times):
            pred = int(p > threshold)

            if pred and not current_active:
                current_active = True
                current_start = ft
            elif (not pred) and current_active:
                periods.append((current_start, ft))
                current_active = False
                current_start = None

    total_time = frame_idx / fps
    if current_active and current_start is not None:
        periods.append((current_start, total_time))

    cap.release()
    return periods


def format_time(seconds):
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


print("\n" + "=" * 60)
print("ПОШУК ТЕСТОВИХ ЗОБРАЖЕНЬ")
print("=" * 60)

apple_files = [os.path.join(apple_dir, f) for f in os.listdir(apple_dir)
               if os.path.isfile(os.path.join(apple_dir, f)) and is_valid_image(os.path.join(apple_dir, f))]

bmw_files = [os.path.join(bmw_dir, f) for f in os.listdir(bmw_dir)
             if os.path.isfile(os.path.join(bmw_dir, f)) and is_valid_image(os.path.join(bmw_dir, f))]

print(f"Знайдено валідних зображень Apple: {len(apple_files)}")
print(f"Знайдено валідних зображень BMW: {len(bmw_files)}")

np.random.shuffle(apple_files)
np.random.shuffle(bmw_files)

test_apple = apple_files[:num_test_images]
test_bmw = bmw_files[:num_test_images]

print(f"\nВибрано для тестування: {len(test_apple)} Apple + {len(test_bmw)} BMW")

test_files = test_apple + test_bmw
true_labels = [1] * len(test_apple) + [0] * len(test_bmw)

combined = list(zip(test_files, true_labels))
np.random.shuffle(combined)
test_files, true_labels = zip(*combined)
test_files = list(test_files)
true_labels = list(true_labels)

if len(test_files) > 0:
    accuracy, predictions = predict_images(model, test_files, true_labels, show_images=True)
else:
    print("\n✗ Не знайдено валідних зображень для тестування!")

if os.path.exists(video_path):
    print("\n" + "=" * 60)
    print("ОБРОБКА ВІДЕО")
    print("=" * 60)
    print(f"Шлях до відео: {video_path}")
    print(f"Поріг класифікації: {prediction_threshold}")

    t0 = time.time()
    periods = process_video(video_path, model, img_size=img_size, batch_size=32, threshold=prediction_threshold)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ ОБРОБКИ ВІДЕО")
    print("=" * 60)

    if periods:
        print(f"\nЗнайдено періодів появи об'єкту (Apple): {len(periods)}\n")

        total_duration = sum(e - s for s, e in periods)

        print(f"{'№':<4} {'Початок':<12} {'Кінець':<12} {'Тривалість':<12}")
        print("-" * 45)

        for i, (start, end) in enumerate(periods, 1):
            duration = end - start
            print(f"{i:<4} {format_time(start):<12} {format_time(end):<12} {duration:>10.2f}s")

        print("-" * 45)
        print(f"{'ВСЬОГО:':<28} {total_duration:>10.2f}s")

        print(f"\nСтатистика:")
        print(f"  Загальна тривалість появи об'єкту: {total_duration:.2f} сек")
        print(f"  Середня тривалість періоду: {total_duration / len(periods):.2f} сек")
        print(f"  Найкоротший період: {min(e - s for s, e in periods):.2f} сек")
        print(f"  Найдовший період: {max(e - s for s, e in periods):.2f} сек")
    else:
        print("\n⚠ Об'єкт не знайдено у відео")

    print(f"\nЧас обробки відео: {elapsed:.1f} сек")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    processing_speed = video_duration / elapsed
    print(f"Швидкість обробки: {processing_speed:.2f}x від реального часу")
    print(f"FPS обробки: {total_frames / elapsed:.2f}")

else:
    print("\n" + "=" * 60)
    print("ВІДЕО НЕ ЗНАЙДЕНО")
    print("=" * 60)
    print(f"Файл не існує: {video_path}")
    print("Пропускаю етап тестування відео.")