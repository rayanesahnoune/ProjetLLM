from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
from werkzeug.security import generate_password_hash, check_password_hash

vocab = {
    "the": 0, "apothecary": 1, "diaries": 2, "pride": 3, "and": 4, "prejudice": 5, "naruto": 6,
    "jojo": 7, "bizarre": 8, "adventures": 9, "is": 10, "a": 11, "anime": 12, "novel": 13, "your": 14,
    "name": 15, "one": 16, "piece": 17, "harry": 18, "potter": 19, "stranger": 20, "an": 21, "best": 22,
    "?": 23, "isn't": 24, "it": 25, "barrier": 26, ",": 27, "of": 28, "not": 29, "great": 30, "classic": 31,
    "famous": 32, "really": 33, "just": 34, "good": 35, "written": 36, "by": 37, "camus": 38, "austen": 39,
    "about": 40, "character": 41, "story": 42, "based": 43, "on": 44, "manga": 45, "series": 46, "book": 47,
    "people": 48, "love": 49,
}

INDEX_TO_WORD = {v: k for k, v in vocab.items()}
SEQUENCE_LENGTH = 9


class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, X, mask=None, **kwargs):
        S = X @ tf.transpose(X, perm=[0, 2, 1])
        if mask is not None:
            m = tf.cast(mask, tf.float32)[:, tf.newaxis, :]
            S = S + (1.0 - m) * -1e9
        seq_len = tf.shape(X)[1]
        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        S = S + (1.0 - causal) * -1e9
        A = tf.nn.softmax(S, axis=-1)
        return A @ X


class MultiHeadSimpleAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.heads = [SimpleAttention() for _ in range(num_heads)]
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, X, mask=None, training=False, **kwargs):
        concat = tf.concat([h(X, mask=mask) for h in self.heads], axis=-1)
        return self.dense(concat)


@tf.keras.utils.register_keras_serializable()
class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.att = MultiHeadSimpleAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length)
        return self.token_embeddings(inputs) + self.position_embeddings(positions)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class SmallGPT(tf.keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
        self.decoder_blocks = [TransformerDecoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.pos_embedding(inputs)
        x._keras_mask = None
        for block in self.decoder_blocks:
            x = block(x, training=training)
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


phrases = [
    "naruto is the best anime isn't it ?",
    "one piece is a really great anime ",
    "jojo bizarre adventures is a great anime ",
    "your name is a famous anime isn't it ?",
    "naruto is a famous anime series isn't it ?",
    "one piece is a classic anime series ",
    "jojo bizarre adventures is the best anime",
    "the apothecary diaries is a great anime",
    "naruto is based on a great manga series",
    "one piece is based on a famous manga",
    "your name is just a really good anime",
    "jojo bizarre adventures is based on manga ",
    "the apothecary diaries is not a novel ",
    "naruto is not a novel it is anime",
    "one piece is not a novel it is anime",
    "pride and prejudice is the best novel ",
    "harry potter is the best novel isn't it ?",
    "the stranger is a classic novel by camus",
    "pride and prejudice is written by austen ",
    "harry potter is a famous novel series ",
    "the stranger is a really good novel ",
    "pride and prejudice is a classic novel ",
    "harry potter is based on a great novel",
    "the stranger is not an anime it is novel",
    "pride and prejudice is not an anime ",
    "harry potter is not anime it is a novel",
    "the stranger is a novel about one character",
    "pride and prejudice is a story about love",
    "harry potter is a novel people really love",
    "the stranger is the best novel by camus",
]


def _tokeniser(phrase, v):
    return [v[mot] for mot in phrase.split() if mot in v]


_X = [_tokeniser(p, vocab) for p in phrases]
_X = _pad(_X, maxlen=10, padding='post')
_split = int(0.8 * len(_X))
_X_train, _X_test = _X[:_split], _X[_split:]
_y_train_lm = _X_train[:, 1:]
_X_train_lm = _X_train[:, :-1]
_y_test_lm = _X_test[:, 1:]
_X_test_lm = _X_test[:, :-1]

MODEL = SmallGPT(SEQUENCE_LENGTH, len(vocab), 32, 2, 64, 1)
MODEL.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
MODEL.fit(
    _X_train_lm,
    _y_train_lm,
    validation_data=(_X_test_lm, _y_test_lm),
    epochs=20,
    batch_size=8,
    verbose=0
)
print("✅ Modèle ITRIA entraîné.")


def generate_with_explanation(prompt_text: str, num_words: int = 5):
    words = prompt_text.lower().split()
    unknown = [w for w in words if w not in vocab]
    tokens = [vocab[w] for w in words if w in vocab]

    if not tokens:
        return {
            "error": True,
            "message": f"Aucun mot du prompt n'est dans le vocabulaire. Mots inconnus : {', '.join(unknown)}"
        }

    steps = []
    generated = []

    for _ in range(num_words):
        seq = tokens[-SEQUENCE_LENGTH:]
        seq_padded = _pad([seq], maxlen=SEQUENCE_LENGTH, padding='post')
        probs = MODEL(seq_padded, training=False).numpy()[0]
        last_pos = min(len(tokens) - 1, SEQUENCE_LENGTH - 1)
        token_probs = probs[last_pos]

        next_idx = int(np.argmax(token_probs))
        next_word = INDEX_TO_WORD.get(next_idx, "?")
        top3_idx = np.argsort(token_probs)[::-1][:3]
        top3 = [
            {
                "word": INDEX_TO_WORD.get(int(i), "?"),
                "prob": round(float(token_probs[i]) * 100, 1)
            }
            for i in top3_idx
        ]

        steps.append({"word": next_word, "top3": top3})
        generated.append(next_word)
        tokens.append(next_idx)

    prompt_clean = " ".join(w for w in words if w in vocab)
    return {
        "error": False,
        "prompt": prompt_clean,
        "generated": generated,
        "full_text": prompt_clean + " " + " ".join(generated),
        "steps": steps,
        "unknown": unknown,
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(BASE_DIR, "..", "Frontend")
FRONTEND = os.path.abspath(FRONTEND)
DATABASE = os.path.join(BASE_DIR, "users.db")
app = Flask(__name__, template_folder=FRONTEND, static_folder=FRONTEND)

app.secret_key = "itria-secret-key-change-me"

DEFAULT_USERS = {
    "yanis": "0000",
    "sylia": "1234",
    "rayane": "123",
    "itria": "itria2024"
}


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_default_users():
    conn = get_db_connection()
    for username, password in DEFAULT_USERS.items():
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if not existing_user:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )

    conn.commit()
    conn.close()


init_db()
create_default_users()


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("chat"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("login.html")

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs."), 400

    conn = get_db_connection()
    user = conn.execute(
        "SELECT username, password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        session["user"] = user["username"]
        return jsonify(success=True), 200

    return jsonify(success=False, error="Identifiant ou mot de passe incorrect."), 401


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if "user" in session:
            return redirect(url_for("chat"))
        return render_template("register.html")

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify(success=False, error="Veuillez remplir tous les champs."), 400

    conn = get_db_connection()
    existing_user = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,)
    ).fetchone()

    if existing_user:
        conn.close()
        return jsonify(success=False, error="Ce nom d'utilisateur existe déjà."), 409

    conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, generate_password_hash(password))
    )
    conn.commit()
    conn.close()

    return jsonify(success=True), 201


@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("chat.html", username=session["user"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify(error=True, message="Non authentifié."), 401

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    num_words = int(data.get("num_words", 5))

    if not prompt:
        return jsonify(error=True, message="Prompt vide."), 400

    result = generate_with_explanation(prompt, num_words)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)