import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D, LSTM, Attention, Input, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class TransactionClassifier:
    def __init__(self, max_words=5000, max_len=15, embed_dim=64, confidence_threshold=0.6):
        self.max_words = max_words
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.confidence_threshold = confidence_threshold

        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.amount_pattern = re.compile(r'\d+')

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
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        return text.strip().lower()

    def _ensure_others_category(self, df):
        if 'others' not in df['category'].unique():
            filler = pd.DataFrame([
                {'text': 'giao dịch lạ 100000', 'clean_text': 'giao dịch lạ', 'category': 'others', 'amount': 100000},
                {'text': 'phí không rõ 50000', 'clean_text': 'phí không rõ', 'category': 'others', 'amount': 50000}
            ])
            df = pd.concat([df, filler], ignore_index=True)
        return df

    def extract_amount(self, text):
      text = text.lower().replace('.', '').replace(',', '')  # normalize separators
      match = re.search(r'(\d+(?:\.\d+)?)(k|tr|triệu|vnd)?', text)

      if match:
          number = float(match.group(1))
          unit = match.group(2)

          if unit == 'k':
              amount = int(number * 1_000)
          elif unit in ['tr', 'triệu']:
              amount = int(number * 1_000_000)
          else:
              amount = int(number)

          formatted = f"{amount:,} VND"  # <-- use commas
          return amount, formatted

      return 0, "0 VND"


    def preprocess(self, df):
        self.tokenizer.fit_on_texts(df['clean_text'])
        sequences = self.tokenizer.texts_to_sequences(df['clean_text'])
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = self.label_encoder.fit_transform(df['category'])
        return X, y

    # def build_model(self):
    #     model = Sequential([
    #         Embedding(self.max_words, self.embed_dim, input_length=self.max_len),
    #         GlobalAveragePooling1D(),
    #         Dense(64, activation='relu'),
    #         Dropout(0.5),
    #         Dense(len(self.label_encoder.classes_), activation='softmax')
    #     ])
    #     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model
    def build_model(self):
      inputs = Input(shape=(self.max_len,))
      x = Embedding(self.max_words, self.embed_dim)(inputs)
      x = Bidirectional(LSTM(64, return_sequences=True))(x)
      x = GlobalAveragePooling1D()(x)
      x = Dense(64, activation='relu')(x)
      x = Dropout(0.5)(x)
      outputs = Dense(len(self.label_encoder.classes_), activation='softmax')(x)

      model = Model(inputs, outputs)
      model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      return model


    def train(self, X, y, epochs=10, batch_size=32):
        self.model = self.build_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def evaluate(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    def predict(self, text):
        if self.model is None:
            raise Exception("Model not trained")

        clean_text = self._clean_text(text)
        seq = self.tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=self.max_len)

        amount, formatted_amount = self.extract_amount(text)
        pred_probs = self.model.predict(padded)[0]
        pred_index = np.argmax(pred_probs)
        confidence = pred_probs[pred_index]
        category = self.label_encoder.inverse_transform([pred_index])[0]

        if confidence < self.confidence_threshold and 'others' in self.label_encoder.classes_:
            category = 'others'

        return category, formatted_amount, float(confidence)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'model_config': self.model.get_config(),
                'model_weights': self.model.get_weights(),
                'params': {
                    'max_words': self.max_words,
                    'max_len': self.max_len,
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
            self.max_len = data['params']['max_len']
            self.embed_dim = data['params']['embed_dim']
            self.confidence_threshold = data['params']['confidence_threshold']
            self.model = Sequential.from_config(data['model_config'])
            self.model.set_weights(data['model_weights'])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def main():
    clf = TransactionClassifier()
    df = clf.load_data("app/train.jsonl")
    X, y = clf.preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining model...")
    clf.train(X_train, y_train, epochs=15, batch_size=32)

    print("\nEvaluating model...")
    clf.evaluate(X_test, y_test)

    clf.save("transaction_classifier.pkl")

    print("\nSample Predictions:")
    test_cases = [
        "ăn nhà hàng 200000",
        "mua sắm tại shop 1,500,000đ",
        "giao dịch không xác định 500000",
        "thanh toán app abc 300000",
        "tiền điện 325.000 đồng",
        "trả tiền khóa học 2500000",
        "mua đồ chơi trẻ em 150000"
    ]

    for text in test_cases:
        category, amount, confidence = clf.predict(text)
        print(f"Text: {text}\n=> Category: {category}, Amount: {amount}, Confidence: {confidence:.2f}\n")


if __name__ == '__main__':
    main()