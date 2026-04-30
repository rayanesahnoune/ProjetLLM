import tensorflow as tf 
class SimpleAttention(tf.keras.layers.Layer):
    """Couche d'attention simple sans projection linéaire.
    Calcule les scores d'attention entre tous les tokens d'une séquence,
    applique un masque causal pour empêcher le modèle de voir les tokens futurs,
    et retourne une représentation pondérée de la séquence.
    
    Attributes:
        S: (tensor) matrice des scores d'attention avant softmax
        A: (tensor) matrice d'attention après softmax (les poids)
        O: (tensor) sortie de la couche
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.S = None
        self.A = None
        self.O = None

    def call(self, X, mask=None, **kwargs):
        """Calcul de l'attention sur X avec masque causal.
            
        Args:
            X: (tensor) séquence d'entrée de shape (batch, seq_len, embed_dim)
            mask: (tensor) masque de padding optionnel de shape (batch, seq_len)
        Returns:
            O: (tensor) sortie de shape identique à X
        """
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
    """Attention multi-têtes basée sur SimpleAttention.
    Exécute plusieurs SimpleAttention en parallèle sur la même entrée,
    chaque tête apprend à se concentrer sur des aspects différents de la séquence.
    Les sorties de toutes les têtes sont concaténées puis réduites
    via une couche Dense pour revenir à la dimension d'origine.

    Attributes:
        num_heads: (int) nombre de têtes d'attention
        embed_dim: (int) dimension de l'embedding
        heads: (list) liste des SimpleAttention
        dense: (Dense) couche de projection finale
    """
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