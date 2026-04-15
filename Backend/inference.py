import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Model"))

sys.path.append(MODEL_DIR)

from smallGPT import SmallGPT

MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "best_smallgpt.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)
TOKENIZER_PATH = os.path.join(BASE_DIR, "saved_model", "tokenizer.pkl")
MAXLEN_PATH = os.path.join(BASE_DIR, "saved_model", "max_sequence_len.npy")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"smallGPT": SmallGPT},
    compile=False
)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = int(np.load(MAXLEN_PATH))

def predict_next_word(prompt):
    token_list = tokenizer.texts_to_sequences([prompt])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)
    
    if len(prediction.shape) == 3:
        predicted_index = np.argmax(prediction[0, -1, :])
    else:
        predicted_index = np.argmax(prediction[0, :])

    predicted_index = int(predicted_index)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""