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


MODEL_PATH = os.path.join(MODEL_DIR, "best_smallgpt_oz.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

FILE_ID = "1Ek-o20d-frKLlQj2NDw5QhGEthpkEClK"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

MODEL_VERSION = "v1"
VERSION_FILE = os.path.join(MODEL_DIR, "model_version.txt")



need_download = True

if os.path.exists(MODEL_PATH) and os.path.exists(VERSION_FILE):
    with open(VERSION_FILE, "r") as f:
        saved_version = f.read().strip()
    if saved_version == MODEL_VERSION:
        need_download = False
        print(f" Modèle déjà à jour ({MODEL_VERSION}), pas de téléchargement.")


if need_download:
    print(f" Téléchargement du modèle vers : {MODEL_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        result = gdown.download(URL, MODEL_PATH, quiet=False)

        if result is None or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"gdown n'a pas pu télécharger le fichier.\n"
                f"  → Vérifiez que le fichier Google Drive est bien public (accès 'Tout le monde avec le lien').\n"
                f"  → FILE_ID utilisé : {FILE_ID}"
            )

        if os.path.getsize(MODEL_PATH) < 1_000:
            raise RuntimeError(
                f"Le fichier téléchargé est trop petit ({os.path.getsize(MODEL_PATH)} octets).\n"
                f"  → Google Drive a probablement renvoyé une page HTML au lieu du fichier.\n"
                f"  → Assurez-vous que le partage est public et que FILE_ID est correct."
            )

        with open(VERSION_FILE, "w") as f:
            f.write(MODEL_VERSION)

        print(f"[inference] Modèle téléchargé avec succès ({os.path.getsize(MODEL_PATH) / 1e6:.1f} Mo).")

    except Exception as e:

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError(f"[inference] Échec du téléchargement du modèle : {e}") from e




model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"smallGPT": SmallGPT},
    compile=False
)

TOKENIZER_PATH = os.path.join(BASE_DIR, "saved_model", "tokenizer_oz.pkl")
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

    return None
