import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings

warnings.filterwarnings('ignore')

from jiwer import wer, cer

import re
from collections import Counter
import textdistance
from spellchecker import SpellChecker


def download_ljspeech():
    data_path = '/kaggle/input/ljspeech-1/LJSpeech-1.1'

    if not os.path.exists(data_path):
        os.system('wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2')
        os.system('tar -xjf LJSpeech-1.1.tar.bz2')
        data_path = 'LJSpeech-1.1'

    return data_path


def load_metadata(data_path):
    metadata_path = os.path.join(data_path, 'metadata.csv')

    data = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:
                filename = parts[0]
                text = parts[2]
                audio_path = os.path.join(data_path, 'wavs', f'{filename}.wav')
                data.append({'audio_path': audio_path, 'text': text})

    df = pd.DataFrame(data)
    print(f"Завантажено {len(df)} аудіофайлів")
    return df


class AudioPreprocessor:

    def __init__(self, sr=16000, n_mfcc=13, n_mels=128, n_fft=512, hop_length=160):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, path):
        audio, sr = librosa.load(path, sr=self.sr)
        return audio

    def extract_features(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)

        return log_mel_spec.T

    def preprocess(self, path):
        audio = self.load_audio(path)
        features = self.extract_features(audio)
        return features


class TextTransform:

    def __init__(self):
        self.chars = [' ', "'"] + list('abcdefghijklmnopqrstuvwxyz')
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.blank_idx = len(self.chars)

    def text_to_int(self, text):
        text = text.lower()
        return [self.char_to_idx.get(char, self.char_to_idx[' ']) for char in text if char in self.char_to_idx]

    def int_to_text(self, indices):
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])

    def vocab_size(self):
        return len(self.chars) + 1


class LJSpeechDataset(Dataset):

    def __init__(self, df, preprocessor, text_transform):
        self.df = df
        self.preprocessor = preprocessor
        self.text_transform = text_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = self.preprocessor.preprocess(row['audio_path'])

        text_indices = self.text_transform.text_to_int(row['text'])

        return (
            torch.FloatTensor(features),
            torch.LongTensor(text_indices),
            features.shape[0],
            len(text_indices)
        )


def collate_fn(batch):
    features, texts, feature_lens, text_lens = zip(*batch)

    features_padded = pad_sequence(features, batch_first=True)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    return (
        features_padded,
        texts_padded,
        torch.LongTensor(feature_lens),
        torch.LongTensor(text_lens)
    )


class DeepSpeech2(nn.Module):
    def __init__(self, n_features, n_classes, n_rnn_layers=5, rnn_hidden=512, dropout=0.1):
        super(DeepSpeech2, self).__init__()

        self.n_features = n_features
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout)
        )

        conv_out_size = self._get_conv_out_size(n_features)

        self.rnn_layers = nn.ModuleList()
        for i in range(n_rnn_layers):
            input_size = conv_out_size if i == 0 else rnn_hidden * 2
            self.rnn_layers.append(
                nn.GRU(input_size, rnn_hidden, num_layers=1,
                       bidirectional=True, batch_first=True, dropout=0)
            )

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(rnn_hidden * 2) for _ in range(n_rnn_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(rnn_hidden * 2, n_classes)

    def _get_conv_out_size(self, n_features):
        with torch.no_grad():
            x = torch.zeros(1, 1, 100, n_features)
            x = self.conv(x)
            conv_out = x.size(1) * x.size(3)
        return conv_out

    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch_size, channels, time, features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, time, channels * features)
        lengths = lengths // 4

        for i, rnn in enumerate(self.rnn_layers):
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            x_packed, _ = rnn(x_packed)
            x, _ = pad_packed_sequence(x_packed, batch_first=True)

            x = x.transpose(1, 2)
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)

            x = self.dropout(x)

        x = self.fc(x)

        x = nn.functional.log_softmax(x, dim=-1)

        x = x.transpose(0, 1)

        return x, lengths


def train_model(model, train_loader, val_loader, text_transform,
                epochs=30, lr=0.0003, device='cuda'):
    model = model.to(device)
    criterion = nn.CTCLoss(blank=text_transform.blank_idx, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_idx, (features, texts, feature_lens, text_lens) in enumerate(train_loader):
            features = features.to(device)
            texts = texts.to(device)

            optimizer.zero_grad()

            outputs, output_lengths = model(features, feature_lens)

            loss = criterion(outputs, texts, output_lengths, text_lens)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for features, texts, feature_lens, text_lens in val_loader:
                features = features.to(device)
                texts = texts.to(device)

                outputs, output_lengths = model(features, feature_lens)
                loss = criterion(outputs, texts, output_lengths, text_lens)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'deepspeech2_best.pth')
            print('Model saved!')

        print('-' * 60)

    return model


class GreedyDecoder:
    def __init__(self, text_transform):
        self.text_transform = text_transform
        self.blank_idx = text_transform.blank_idx

    def decode(self, output):
        arg_maxes = torch.argmax(output, dim=-1)

        decoded_texts = []
        for i in range(arg_maxes.size(1)):
            indices = arg_maxes[:, i].cpu().numpy()

            decoded = []
            prev_idx = None
            for idx in indices:
                if idx != self.blank_idx and idx != prev_idx:
                    decoded.append(idx)
                prev_idx = idx

            text = self.text_transform.int_to_text(decoded)
            decoded_texts.append(text)

        return decoded_texts


def test_model(model, test_loader, text_transform, device='cuda'):
    model = model.to(device)
    model.eval()

    decoder = GreedyDecoder(text_transform)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, texts, feature_lens, text_lens in test_loader:
            features = features.to(device)

            outputs, output_lengths = model(features, feature_lens)

            predictions = decoder.decode(outputs)

            for i in range(texts.size(0)):
                target_indices = texts[i, :text_lens[i]].cpu().numpy()
                target_text = text_transform.int_to_text(target_indices)
                all_targets.append(target_text)

            all_predictions.extend(predictions)

    wer_score = wer(all_targets, all_predictions)
    cer_score = cer(all_targets, all_predictions)

    print(f'\nTest Results:')
    print(f'WER: {wer_score * 100:.2f}%')
    print(f'CER: {cer_score * 100:.2f}%')

    print('\nПриклади розпізнавання:')
    for i in range(min(5, len(all_predictions))):
        print(f'\nTarget:     {all_targets[i]}')
        print(f'Prediction: {all_predictions[i]}')

    return all_predictions, all_targets, wer_score


class TextCorrector:
    def __init__(self):
        self.spell = SpellChecker()
        self.word_freq = Counter()

    def train_on_corpus(self, texts):
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)

    def correct_spelling(self, text):
        words = text.split()
        corrected = []

        for word in words:
            if word in self.spell:
                corrected.append(word)
            else:
                correction = self.spell.correction(word)
                corrected.append(correction if correction else word)

        return ' '.join(corrected)

    def correct_with_edit_distance(self, text, vocabulary, max_distance=2):
        words = text.split()
        corrected = []

        for word in words:
            if word in vocabulary:
                corrected.append(word)
            else:
                best_match = min(vocabulary,
                                 key=lambda v: textdistance.levenshtein(word, v))

                if textdistance.levenshtein(word, best_match) <= max_distance:
                    corrected.append(best_match)
                else:
                    corrected.append(word)

        return ' '.join(corrected)

    def correct_with_frequency(self, text):
        words = text.split()
        corrected = []

        for word in words:
            if word in self.word_freq:
                corrected.append(word)
            else:
                candidates = [w for w in self.word_freq.keys()
                              if textdistance.levenshtein(word, w) <= 2]

                if candidates:
                    best = max(candidates, key=lambda w: self.word_freq[w])
                    corrected.append(best)
                else:
                    corrected.append(word)

        return ' '.join(corrected)

    def correct_regex_patterns(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        return text


def compare_correction_methods(predictions, targets, corrector, vocabulary):
    print("\n" + "=" * 80)
    print("ПОРІВНЯННЯ МЕТОДІВ ВИПРАВЛЕННЯ ПОМИЛОК")
    print("=" * 80)

    methods = {
        'Без виправлення': predictions,
        'Spell Checker': [corrector.correct_spelling(pred) for pred in predictions],
        'Edit Distance': [corrector.correct_with_edit_distance(pred, vocabulary) for pred in predictions],
        'Frequency Based': [corrector.correct_with_frequency(pred) for pred in predictions],
        'Regex Patterns': [corrector.correct_regex_patterns(pred) for pred in predictions]
    }

    results = {}

    for method_name, corrected_texts in methods.items():
        wer_score = wer(targets, corrected_texts)
        cer_score = cer(targets, corrected_texts)

        results[method_name] = {
            'WER': wer_score * 100,
            'CER': cer_score * 100
        }

        print(f"\n{method_name}:")
        print(f"  WER: {wer_score * 100:.2f}%")
        print(f"  CER: {cer_score * 100:.2f}%")

    best_method = min(results.items(), key=lambda x: x[1]['WER'])
    print(f"\n{'=' * 80}")
    print(f"Найкращий метод: {best_method[0]} (WER: {best_method[1]['WER']:.2f}%)")
    print(f"{'=' * 80}")

    return results


def main():
    print("=" * 80)
    print("DEEPSPEECH2 ДЛЯ LJ-SPEECH DATASET")
    print("=" * 80)

    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 0.0003
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nВикористовується пристрій: {DEVICE}")

    data_path = download_ljspeech()
    df = load_metadata(data_path)

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n" + "-" * 80)
    print("2. ІНІЦІАЛІЗАЦІЯ ПРЕПРОЦЕСОРА")
    print("-" * 80)
    preprocessor = AudioPreprocessor()
    text_transform = TextTransform()

    print(f"Розмір словника: {text_transform.vocab_size()}")

    print("\n" + "-" * 80)
    print("3. СТВОРЕННЯ DATASETS")
    print("-" * 80)
    train_dataset = LJSpeechDataset(train_df, preprocessor, text_transform)
    val_dataset = LJSpeechDataset(val_df, preprocessor, text_transform)
    test_dataset = LJSpeechDataset(test_df, preprocessor, text_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn, num_workers=2)

    print("\n" + "-" * 80)
    print("4. СТВОРЕННЯ МОДЕЛІ DEEPSPEECH2")
    print("-" * 80)
    model = DeepSpeech2(
        n_features=128,
        n_classes=text_transform.vocab_size(),
        n_rnn_layers=5,
        rnn_hidden=512,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Загальна кількість параметрів: {total_params:,}")

    print("\n" + "-" * 80)
    print("5. НАВЧАННЯ МОДЕЛІ")
    print("-" * 80)
    model = train_model(model, train_loader, val_loader, text_transform,
                        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)

    print("\n" + "-" * 80)
    print("6. ТЕСТУВАННЯ МОДЕЛІ")
    print("-" * 80)

    model.load_state_dict(torch.load('deepspeech2_best.pth'))

    predictions, targets, wer_score = test_model(
        model, test_loader, text_transform, device=DEVICE
    )

    print("\n" + "-" * 80)
    print("7. ДОДАТКОВЕ ЗАВДАННЯ: ВИПРАВЛЕННЯ ПОМИЛОК")
    print("-" * 80)

    corrector = TextCorrector()

    print("Навчання corrector на train corpus...")
    corrector.train_on_corpus(train_df['text'].tolist())

    vocabulary = set(' '.join(train_df['text'].tolist()).lower().split())

    results = compare_correction_methods(predictions, targets, corrector, vocabulary)

    print("\n" + "=" * 80)
    print("ВИКОНАННЯ ЗАВЕРШЕНО!")
    print("=" * 80)

    return model, results


if __name__ == "__main__":
    main()