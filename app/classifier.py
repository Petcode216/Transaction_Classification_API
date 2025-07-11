import re
import json
import random
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Embedding, GlobalAveragePooling1D,
    Bidirectional, LSTM, Dense, Dropout
)

class TransactionClassifier:
    def __init__(
        self,
        max_words=5000,
        max_len=15,
        embed_dim=64,
        confidence_threshold=0.6
    ):
        self.max_words = max_words
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.confidence_threshold = confidence_threshold

        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None

        # Mở rộng regex: nhóm 1 = dấu [+/-]?, nhóm 2 = số, nhóm 3 = đơn vị?
        self.amount_pattern = re.compile(r'([+\-]?)(\d+(?:\.\d+)?)(k|tr|triệu|vnd)?', re.IGNORECASE)

    def load_data(self, filepath):
        records = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    raw = json.loads(line)
                    output = json.loads(raw['output'])
                    records.append({
                        'text': raw['text'],
                        'clean_text': self._clean_text(raw['text']),
                        'category': output['category'],
                        'amount': output['amount']
                    })
                except:
                    continue
        df = pd.DataFrame(records)
        return self._ensure_others_category(df)

    def _clean_text(self, text):
        # Loại bỏ toàn bộ chữ số và ký tự đặc biệt (bao gồm +, -) trước khi training
        text = re.sub(r'[+\-]?\d+(?:[.,]\d+)?(?:k|tr|triệu|vnd)?', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        return text.lower().strip()

    def _ensure_others_category(self, df):
        if 'others' not in df['category'].unique():
            filler = pd.DataFrame([
                {'text': 'giao dịch lạ 100000', 'clean_text': 'giao dịch lạ', 'category': 'others', 'amount': 100000},
                {'text': 'phí không rõ 50000',  'clean_text': 'phí không rõ', 'category': 'others', 'amount':  50000}
            ])
            df = pd.concat([df, filler], ignore_index=True)
        return df

    def extract_amount(self, text):
        """
        Trả về:
          - amt (int): giá trị nguyên, có dấu +/−
          - formatted (str): ví dụ "+3,000 VND" hoặc "-2,691,000 VND"
        """
        s = text.replace('.', '').replace(',', '').lower()
        m = self.amount_pattern.search(s)
        if not m:
            return 0, "0 VND"

        sign_char, num_str, unit = m.groups()
        sign = -1 if sign_char == '-' else 1
        number = float(num_str)

        if unit == 'k':
            amt = int(number * 1_000)
        elif unit in ('tr', 'triệu'):
            amt = int(number * 1_000_000)
        else:
            amt = int(number)

        amt *= sign
        prefix = '+' if amt >= 0 else '-'
        formatted = f"{prefix}{abs(amt):,} VND"
        return amt, formatted


    def preprocess(self, df):
        self.tokenizer.fit_on_texts(df['clean_text'])
        seqs = self.tokenizer.texts_to_sequences(df['clean_text'])
        X = pad_sequences(seqs, maxlen=self.max_len)
        y = self.label_encoder.fit_transform(df['category'])
        return X, y

    def build_model(self):
        inputs = Input(shape=(self.max_len,))
        x = Embedding(self.max_words, self.embed_dim)(inputs)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(len(self.label_encoder.classes_), activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        self.model = self.build_model()
        self.model.fit(X, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.2)

    def evaluate(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

    def predict(self, text):
        if self.model is None:
            raise Exception("Model chưa được train")

        clean = self._clean_text(text)
        seq = self.tokenizer.texts_to_sequences([clean])
        pad = pad_sequences(seq, maxlen=self.max_len)

        amt, _ = self.extract_amount(text)     # <-- chỉ lấy phần int
        probs = self.model.predict(pad)[0]
        idx = np.argmax(probs)
        conf = float(probs[idx])
        cat = self.label_encoder.inverse_transform([idx])[0]

        if conf < self.confidence_threshold and 'others' in self.label_encoder.classes_:
            cat = 'others'

        return cat, amt, conf           # <-- trả về amt là int


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'model_config':  self.model.get_config(),
                'model_weights': self.model.get_weights(),
                'params': {
                    'max_words': self.max_words,
                    'max_len':   self.max_len,
                    'embed_dim': self.embed_dim,
                    'confidence_threshold': self.confidence_threshold
                }
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.tokenizer = data['tokenizer']
        self.label_encoder = data['label_encoder']
        self.max_words = data['params']['max_words']
        self.max_len   = data['params']['max_len']
        self.embed_dim = data['params']['embed_dim']
        self.confidence_threshold = data['params']['confidence_threshold']

        self.model = Sequential.from_config(data['model_config'])
        self.model.set_weights(data['model_weights'])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )