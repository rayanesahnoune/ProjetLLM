import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Model"))

sys.path.append(MODEL_DIR)

from smallGPT import SmallGPT


MODEL_PATH = os.path.join(MODEL_DIR, "best_smallgpt.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

FILE_ID = "1gjaqX4WS_-uJ3dr-PjiPiDHklBKrduPG"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

MODEL_VERSION = "v1"
VERSION_FILE = os.path.join(MODEL_DIR, "model_version.txt")



need_download = True

if os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        saved_version = f.read().strip()
        if saved_version == MODEL_VERSION:
            need_download = False

if need_download:
    print("🔄 Téléchargement / mise à jour du modèle...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    gdown.download(URL, MODEL_PATH, quiet=False)
    with open(VERSION_FILE, "w") as f:
        f.write(MODEL_VERSION)

    print("✅ Modèle prêt.")



model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"smallGPT": SmallGPT},
    compile=False
)



TOKENIZER_PATH = os.path.join(BASE_DIR, "saved_model", "tokenizer.pkl")
MAXLEN_PATH = os.path.join(BASE_DIR, "saved_model", "max_sequence_len.npy")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_len = int(np.load(MAXLEN_PATH))

def predict_next_words(prompt):
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