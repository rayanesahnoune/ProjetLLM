"""
Script d'entraînement et de sauvegarde du modèle ITRIA.
Lance ce script UNE FOIS avant de démarrer app.py :
    python Model/train_and_save.py
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    def get_config(self):
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

    def get_config(self):
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


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

phrases = [
    "naruto is the best anime isn't it ?",
    "one piece is a really great anime ,",
    "jojo bizarre adventures is a great anime ,",
    "your name is a famous anime isn't it ?",
    "naruto is a famous anime series isn't it ?",
    "one piece is a classic anime series ,",
    "jojo bizarre adventures is the best anime",
    "the apothecary diaries is a great anime",
    "naruto is based on a great manga series",
    "one piece is based on a famous manga",
    "your name is just a really good anime",
    "jojo bizarre adventures is based on manga ,",
    "the apothecary diaries is not a novel ,",
    "naruto is not a novel it is anime",
    "one piece is not a novel it is anime",
    "pride and prejudice is the best novel ,",
    "harry potter is the best novel isn't it ?",
    "the stranger is a classic novel by camus",
    "pride and prejudice is written by austen ,",
    "harry potter is a famous novel series ,",
    "the stranger is a really good novel ,",
    "pride and prejudice is a classic novel ,",
    "harry potter is based on a great novel",
    "the stranger is not an anime it is novel",
    "pride and prejudice is not an anime ,",
    "harry potter is not anime it is a novel",
    "the stranger is a novel about one character",
    "pride and prejudice is a story about love",
    "harry potter is a novel people really love",
    "the stranger is the best novel by camus",
]


def tokeniser(phrase, vocab):
    mots = phrase.split()
    return [vocab[mot] for mot in mots if mot in vocab]


X_sequences, y_labels = [], []
for phrase in phrases:
    tokens = tokeniser(phrase, vocab)
    for i in range(1, len(tokens)):
        X_sequences.append(tokens[:i])
        y_labels.append(tokens[i])

X = pad_sequences(X_sequences, maxlen=10, padding='pre')
y = np.array(y_labels)

split = int(0.9 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── Construction du modèle ─────────────────────────────────────────────
vocab_size = len(vocab) + 1
embed_dim  = 16
num_heads  = 2
ff_dim     = 32

inputs = tf.keras.layers.Input(shape=(10,))
x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Entraînement ──────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=4,
    validation_data=(X_test, y_test)
)

# ── Sauvegarde au bon endroit ──────────────────────────────────────────
save_path = os.path.join(os.path.dirname(__file__), 'model_itria.keras')
model.save(save_path)
print(f"✅ Modèle sauvegardé : {save_path}")
