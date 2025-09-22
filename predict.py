import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
MODEL_PATH = "inception_transfer.h5"
CLASS_INDICES_PATH = "class_indices.json"
METADATA_PATH = "model_metadata.json"
IMG_SIZE = (299, 299)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr
class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "class_names" in data:
                self.class_names = data["class_names"]
                self.class_indices = data["class_indices"]
            elif isinstance(data, list):
                self.class_names = data
                sorted_names = sorted(data)
                self.class_indices = {name: i for i, name in enumerate(sorted_names)}
            else:
                self.class_indices = data
                inv = {v: k for k, v in data.items()}
                self.class_names = [inv[i] for i in sorted(inv.keys())]
        else:
            self.class_names = ["BMW", "Mercedes"]
            self.class_indices = {"BMW": 0, "Mercedes": 1}
        self.threshold = 0.5
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "threshold" in meta:
                    self.threshold = float(meta["threshold"])
    def predict_single_image(self, img_path):
        arr = preprocess_image(img_path)
        prob = self.model.predict(arr, verbose=0).ravel()[0]

        if prob > self.threshold:
            predicted = self.class_names[1]
            confidence = prob * 100
        else:
            predicted = self.class_names[0]
            confidence = (1 - prob) * 100
        return predicted, confidence
    def evaluate_test_set(self, test_dir, show_examples=True, examples_per_class=5):
        y_true, y_pred, y_prob = [], [], []
        examples_by_true_class = {cls: [] for cls in self.class_names}
        has_subdirs = any(os.path.isdir(os.path.join(test_dir, d)) for d in os.listdir(test_dir))
        if has_subdirs:
            for class_name in self.class_names:
                class_dir = os.path.join(test_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"⚠️ Папка {class_name} не знайдена")
                    continue
                print(f"📁 Обробка класу: {class_name}")
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        fpath = os.path.join(class_dir, fname)
                        arr = preprocess_image(fpath)
                        prob = self.model.predict(arr, verbose=0).ravel()[0]
                        if prob > self.threshold:
                            pred_label = self.class_names[1]
                            conf = prob * 100
                        else:
                            pred_label = self.class_names[0]
                            conf = (1 - prob) * 100
                        y_true.append(class_name)
                        y_pred.append(pred_label)
                        y_prob.append(prob)
                        if len(examples_by_true_class[class_name]) < examples_per_class:
                            examples_by_true_class[class_name].append(
                                {'path': fpath, 'true_class': class_name, 'pred_class': pred_label, 'confidence': conf,
                                 'raw_prob': prob})
        else:
            print("🔍 Папки класів не знайдено, обробляємо всі файли без меток")
            for fname in os.listdir(test_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fpath = os.path.join(test_dir, fname)
                    arr = preprocess_image(fpath)
                    prob = self.model.predict(arr, verbose=0).ravel()[0]
                    if prob > self.threshold:
                        pred_label = self.class_names[1]
                        conf = prob * 100
                    else:
                        pred_label = self.class_names[0]
                        conf = (1 - prob) * 100
                    y_true.append("unknown")
                    y_pred.append(pred_label)
                    y_prob.append(prob)
        if not y_true:
            print("❌ Тестовий набір порожній або неправильно структурований")
            return None
        if "unknown" not in y_true:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, labels=self.class_names, average="weighted", zero_division=0)
            rec = recall_score(y_true, y_pred, labels=self.class_names, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=self.class_names, average="weighted", zero_division=0)
            print("📊 Результати оцінки:")
            print(f"🎯 Accuracy: {acc:.4f}")
            print(f"📈 Precision: {prec:.4f}")
            print(f"📈 Recall: {rec:.4f}")
            print(f"📈 F1-Score: {f1:.4f}")
            cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
            print("🧾 Матриця невідповідностей:")
            print(cm)
            print("📋 Звіт класифікації:")
            print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
            plt.figure(figsize=(6, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Матриця невідповідностей')
            plt.colorbar()
            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, self.class_names, rotation=45)
            plt.yticks(tick_marks, self.class_names)
            plt.ylabel('Істинні')
            plt.xlabel('Прогнозовані')
            plt.tight_layout()
            plt.savefig("test_confusion_matrix.png")
            print("💾 Матриця невідповідностей збережена в test_confusion_matrix.png")
        else:
            acc = None
            print("ℹ️ Істинні метки класів невідомі, метрики не обчислюються")
        if show_examples and has_subdirs:
            self.show_examples_by_true_class(examples_by_true_class)
        return {"accuracy": acc}
    def show_examples_by_true_class(self, examples_by_true_class):
        num_classes = len(self.class_names)
        examples_per_class = max(len(examples) for examples in examples_by_true_class.values())
        if examples_per_class == 0:
            print("❌ Немає прикладів для відображення")
            return
        plt.figure(figsize=(5 * examples_per_class, 5 * num_classes))
        for cls_idx, class_name in enumerate(self.class_names):
            examples = examples_by_true_class[class_name]
            print(f"\n📁 Істинний клас: {class_name}")
            for i, example in enumerate(examples):
                plt_idx = cls_idx * examples_per_class + i + 1
                plt.subplot(num_classes, examples_per_class, plt_idx)
                img = image.load_img(example['path'], target_size=(150, 150))
                plt.imshow(img)
                is_correct = example['true_class'] == example['pred_class']
                status = "✅" if is_correct else "❌"
                title = f"{status} Істина: {example['true_class']}\n"
                title += f"Передбачено: {example['pred_class']}\n"
                title += f"Впевненість: {example['confidence']:.1f}%"
                plt.title(title, fontsize=8)
                plt.axis("off")
                print(
                    f"  📷 {os.path.basename(example['path'])}: передбачено {example['pred_class']} ({example['confidence']:.1f}%) {status}")
        plt.suptitle("Приклади по істинним класам (з папок)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("examples_by_true_classes.png", dpi=150, bbox_inches='tight')
        print("\n💾 Збережено: examples_by_true_classes.png")
        print("📊 В кожному рядку показані зображення з відповідної ПАПКИ")
if __name__ == "__main__":
    predictor = Predictor()
    test_image = input("Введіть шлях до тестового зображення (або Enter для пропуску): ").strip()
    if test_image and os.path.exists(test_image):
        pred, conf = predictor.predict_single_image(test_image)
        print(f"🔮 Результат: {pred} ({conf:.1f}%)")
    test_dir = "dataset"
    if os.path.exists(test_dir):
        print(f"\n📊 Оцінка на датасеті {test_dir}")
        predictor.evaluate_test_set(test_dir, show_examples=True, examples_per_class=5)
    else:
        print("ℹ️ Папка dataset не знайдена. Додайте туди зображення для перевірки.")