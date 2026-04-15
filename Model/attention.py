import tensorflow as tf 
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.S = None
        self.A = None
        self.O = None

    def call(self, X, mask=None, **kwargs):
        S = X @ tf.transpose(X, perm=[0, 2, 1])


        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = mask[:, tf.newaxis, :]
            S = S + (1.0 - mask) * -1e9


        seq_len = tf.shape(X)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        S = S + (1.0 - causal_mask) * -1e9

        A = tf.nn.softmax(S, axis=-1)
        O = A @ X

        self.S = S
        self.A = A
        self.O = O
        return O

    def get_S(self): return self.S
    def get_A(self): return self.A
    def get_O(self): return self.O
class MultiHeadSimpleAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.heads = [SimpleAttention() for _ in range(num_heads)]
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, X, mask=None, training=False, **kwargs):

        outputs = [head(X, mask=mask) for head in self.heads]

        concat = tf.concat(outputs, axis=-1)
        return self.dense(concat)