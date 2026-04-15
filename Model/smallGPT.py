
import tensorflow as tf
from decoder import PositionalEmbedding, TransformerDecoderBlock

@tf.keras.utils.register_keras_serializable()
class SmallGPT(tf.keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim,
                 num_heads, ff_dim, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size= vocab_size
        self.embed_dim = embed_dim
        self.num_heads= num_heads
        self.ff_dim  = ff_dim
        self.num_layers= num_layers

        self.pos_embedding  = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
        self.decoder_blocks = [
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout= tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        x = self.pos_embedding(inputs)
        x._keras_mask = None
        for block in self.decoder_blocks:
            x = block(x, training=training)
        x = self.layernorm(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)
    #Renvoi la configuration du modèle sous forme d'un dictionnaire python contenat l'information nécessaire pour ré instancier le modéle
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size":self.vocab_size,
            "embed_dim":self.embed_dim,
            "num_heads":self.num_heads,
            "ff_dim":self.ff_dim,
            "num_layers":self.num_layers,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)