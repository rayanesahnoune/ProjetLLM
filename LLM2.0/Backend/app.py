from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Chemin vers le Frontend ─────────────────────────────────────────────
FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'Frontend')

app = Flask(__name__, template_folder=FRONTEND)
app.secret_key = "itria-secret-key"  # ← change en prod

# ── Vocabulaire ────────────────────────────────────────────────────────
vocab = {
    "the":0, "apothecary":1, "diaries":2, "pride":3, "and":4, "prejudice":5, "naruto":6,
    "jojo":7, "bizarre":8, "adventures":9, "is":10, "a":11, "anime":12, "novel":13, "your":14,
    "name":15, "one":16, "piece":17, "harry":18, "potter":19, "stranger":20, "an":21, "best":22,
    "?":23, "isn't":24, "it":25, "barrier":26, ",":27, "of":28, "not":29, "great":30, "classic":31,
    "famous":32, "really":33, "just":34, "good":35, "written":36, "by":37, "camus":38, "austen":39,
    "about":40, "character":41, "story":42, "based":43, "on":44, "manga":45, "series":46, "book":47,
    "people":48, "love":49,
}
index_to_word = {v: k for k, v in vocab.items()}

# ── Couches custom ─────────────────────────────────────────────────────
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A = self.O = self.S = None

    def call(self, X, **kwargs):
        S = X @ tf.transpose(X, perm=[0, 2, 1])
        A = tf.nn.softmax(S, axis=-1)
        O = A @ X
        self.A, self.O, self.S = A, O, S
        return O

    def get_config(self):  # ✅ FIX
        return super().get_config()


class MultiHeadSimpleAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.heads = [SimpleAttention() for _ in range(num_heads)]
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, X, training=False, **kwargs):
        outputs = [head(X) for head in self.heads]
        concat = tf.concat(outputs, axis=-1)
        return self.dense(concat)

    def get_config(self):  # ✅ FIX
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "embed_dim": self.embed_dim})
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.att = MultiHeadSimpleAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, X, training=False, **kwargs):
        attn_output = self.att(X)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(X + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):  # ✅ FIX
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ── Chargement du modèle ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Model', 'model_itria.keras')

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        "TransformerBlock": TransformerBlock,
        "MultiHeadSimpleAttention": MultiHeadSimpleAttention,
        "SimpleAttention": SimpleAttention,
    })
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"⚠️  Impossible de charger le modèle : {e}")
    print("   → Lance d'abord : python Model/train_and_save.py")


# ── Prédiction ──────────────────────────────────────────────────────────
def tokeniser(phrase, vocab):
    mots = phrase.lower().split()
    return [vocab[mot] for mot in mots if mot in vocab]


def sample_with_temperature(probs, temperature=0.8, top_k=10):
    """Échantillonnage avec température + top-k pour éviter les répétitions."""
    probs = np.array(probs, dtype=np.float64)
    # Appliquer la température
    probs = np.log(probs + 1e-10) / temperature
    probs = np.exp(probs - np.max(probs))
    # Top-k : garder seulement les k meilleurs tokens
    if top_k > 0:
        top_k_indices = np.argsort(probs)[-top_k:]
        mask = np.zeros_like(probs)
        mask[top_k_indices] = 1
        probs = probs * mask
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def prediction_mot_suivant(phrase, nb_mots=5):
    last_token = None
    for _ in range(nb_mots):
        tokens = tokeniser(phrase, vocab)
        if not tokens:
            break
        tokens_padded = pad_sequences([tokens], maxlen=10, padding='pre')
        prediction = model.predict(tokens_padded, verbose=0)[0]

        # Pénaliser le dernier token pour éviter les répétitions immédiates
        if last_token is not None and last_token < len(prediction):
            prediction[last_token] *= 0.1

        index_gagnant = sample_with_temperature(prediction, temperature=0.8, top_k=10)
        mot_trouve = index_to_word.get(index_gagnant, "?")
        phrase += " " + mot_trouve
        last_token = index_gagnant

        # Stopper sur les tokens de fin de phrase
        if mot_trouve in ("?", "."):
            break
    return phrase


# ── Utilisateurs ────────────────────────────────────────────────────────
USERS = {
    "yanis": "0000",
    "sylia": "1234",
    "rayane": "123",
    "itria": "itria"
}

# ── Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")

    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs.")

    if USERS.get(username) == password:
        session["user"] = username
        return jsonify(success=True)

    return jsonify(success=False, error="Identifiant ou mot de passe incorrect.")


@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["user"])


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify(error="Non autorisé"), 401

    if model is None:
        return jsonify(response="⚠️ Le modèle n'est pas chargé. Lance d'abord : python Model/train_and_save.py")

    data   = request.get_json(force=True)
    phrase = data.get("message", "").strip()

    if not phrase:
        return jsonify(error="Message vide"), 400

    try:
        nb_mots = int(data.get("nb_mots", 5))
        resultat = prediction_mot_suivant(phrase, nb_mots)
        return jsonify(response=resultat)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
