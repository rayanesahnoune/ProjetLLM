import os
import re
import pickle
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class PrepareData:
    def __init__(self, sequence_length=20, vocab_size=5000,
                 validation_split=0.1, batch_size=32):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.tokenizer = None
        self.index_to_word = None

    def load_from_url(self, url, start_marker=None, end_marker=None):
        response = requests.get(url)
        text = response.text
        if start_marker:
            start = text.find(start_marker)
            text = text[start:]
        if end_marker:
            end = text.find(end_marker)
            text = text[:end]
        return text

    def load_from_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def clean(self, text):
        text = text.replace("\r", "").replace("\n", " ")
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def build(self, text):
        text_clean = self.clean(text)

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.tokenizer.fit_on_texts([text_clean])

        self.vocab_size = min(len(self.tokenizer.word_index) + 1, self.vocab_size)
        self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}

        sequences = self.tokenizer.texts_to_sequences([text_clean])[0]

        X, y = [], []
        for i in range(len(sequences) - self.sequence_length):
            seq = sequences[i:i + self.sequence_length]
            target = sequences[i + 1:i + self.sequence_length + 1]
            X.append(seq)
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        assert X.shape == y.shape, f"Shape mismatch: {X.shape} vs {y.shape}"

        print(f"Vocabulaire   : {self.vocab_size} mots")
        print(f"Séquences     : {len(X)}")
        print(f"X shape       : {X.shape}")
        print(f"y shape       : {y.shape}")

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        dataset_size = sum(1 for _ in dataset)
        val_size = int(self.validation_split * dataset_size)
        train_ds = dataset.skip(val_size)
        val_ds = dataset.take(val_size)

        print(f"Batches train : {sum(1 for _ in train_ds)}")
        print(f"Batches val   : {sum(1 for _ in val_ds)}")

        return train_ds, val_ds

    def decode(self, sequence):
        return " ".join(self.index_to_word.get(int(i), "?") for i in sequence)

    def save_text(self, text, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Texte sauvegardé : {path}")

    def save_artifacts(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        tokenizer_path = os.path.join(output_dir, "tokenizer.pkl")
        maxlen_path = os.path.join(output_dir, "max_sequence_len.npy")

        if self.tokenizer is None:
            raise ValueError("Tokenizer non initialisé. Appelle build(text) avant save_artifacts().")

        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        np.save(maxlen_path, self.sequence_length + 1)

        print(f"Tokenizer sauvegardé : {tokenizer_path}")
        print(f"max_sequence_len sauvegardé : {maxlen_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    prep = PrepareData(sequence_length=20, vocab_size=5000)

    # Remplace ce nom par ton vrai fichier texte d'entraînement
    text_path = os.path.join(BASE_DIR, "data.txt")

    text = prep.load_from_file(text_path)
    train_ds, val_ds = prep.build(text)

    prep.save_artifacts(os.path.join(BASE_DIR, "saved_model"))