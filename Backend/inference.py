"""
inference.py
------------
Charge le modèle SmallGPT une seule fois (lazy loading) et expose
la fonction predict_next_word(prompt) utilisée par app.py.

Structure attendue du projet :
    ProjetLLM/
    ├── Model/
    │   ├── attention.py
    │   ├── decoder.py
    │   ├── smallGPT.py
    │   └── best_smallgpt.keras
    ├── Backend/
    │   ├── app.py
    │   ├── inference.py
    │   └── saved_model/
    │       ├── tokenizer.pkl
    │       └── max_sequence_len.npy
    └── Frontend/
        └── chat.html
"""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# 1. CHEMINS — tout est calculé depuis l'emplacement de ce fichier
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Backend
PROJECT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))    # .../ProjetLLM
MODEL_DIR   = os.path.join(PROJECT_DIR, "Model")                   # .../Model

# Ajoute le dossier Model au path Python pour pouvoir importer smallGPT, etc.
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

MODEL_PATH     = os.path.join(MODEL_DIR,    "best_smallgpt.keras")
TOKENIZER_PATH = os.path.join(BACKEND_DIR,  "saved_model", "tokenizer.pkl")
MAXLEN_PATH    = os.path.join(BACKEND_DIR,  "saved_model", "max_sequence_len.npy")

# ---------------------------------------------------------------------------
# 2. IMPORTS DES CLASSES CUSTOM — après avoir mis à jour sys.path
# ---------------------------------------------------------------------------
from smallGPT import SmallGPT
from decoder  import PositionalEmbedding, TransformerDecoderBlock
from attention import SimpleAttention, MultiHeadSimpleAttention

# Dictionnaire des objets custom à passer à load_model
CUSTOM_OBJECTS = {
    "SmallGPT":                SmallGPT,
    "PositionalEmbedding":     PositionalEmbedding,
    "TransformerDecoderBlock": TransformerDecoderBlock,
    "SimpleAttention":         SimpleAttention,
    "MultiHeadSimpleAttention":MultiHeadSimpleAttention,
}

# ---------------------------------------------------------------------------
# 3. LAZY LOADING — le modèle n'est chargé qu'au premier appel
#    Cela évite de bloquer le démarrage de Flask si le fichier est absent.
# ---------------------------------------------------------------------------
_model     = None
_tokenizer = None
_maxlen    = None


def _load_artifacts():
    """Charge modèle + tokenizer + maxlen une seule fois."""
    global _model, _tokenizer, _maxlen

    if _model is not None:
        return  # déjà chargé

    # Vérifications explicites pour des messages d'erreur clairs
    for path, label in [
        (MODEL_PATH,     "Modèle (.keras)"),
        (TOKENIZER_PATH, "Tokenizer (.pkl)"),
        (MAXLEN_PATH,    "max_sequence_len (.npy)"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} introuvable : {path}")

    print("[inference] Chargement du modèle...")
    _model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False          # pas besoin de recompiler pour l'inférence
    )

    print("[inference] Chargement du tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    _maxlen = int(np.load(MAXLEN_PATH))
    print(f"[inference] Prêt — max_sequence_len={_maxlen}")


# ---------------------------------------------------------------------------
# 4. FONCTION PRINCIPALE
# ---------------------------------------------------------------------------
def predict_next_word(prompt):
    
    token_list = tokenizer.texts_to_sequences([prompt])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    prediction = model.predict(token_list, verbose=0)
    
    if len(prediction.shape) == 3:
        predicted_index = np.argmax(prediction[0, -1, :])
    else:
        predicted_index = np.argmax(prediction[0, :])

    # Forcer la conversion en entier
    predicted_index = int(predicted_index)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return ""