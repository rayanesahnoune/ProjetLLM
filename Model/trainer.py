# trainer.py
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from SmallGPT import SmallGPT
from decoder import PositionalEmbedding, TransformerDecoderBlock
from attention import SimpleAttention, MultiHeadSimpleAttention

class Trainer:
    def __init__(self, model, model_path, epochs=60, patience=5):
        self.model = model
        self.model_path = model_path
        self.epochs= epochs
        self.patience= patience

    def train(self, train_ds, val_ds):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        callbacks = [
            #pour arrêter l'entrainement si ça ne s'améliore pas 
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            #pour enregistrer le modèle à la fin de chaque epoch si le résultat est le meilleur obtenu
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def load(self):
        self.model = load_model(
            self.model_path,
            custom_objects={
                "SimpleAttention": SimpleAttention,
                "MultiHeadSimpleAttention": MultiHeadSimpleAttention,
                "TransformerDecoderBlock":  TransformerDecoderBlock,
                "PositionalEmbedding": PositionalEmbedding,
                "SmallGPT": SmallGPT,
            }
        )
        print(" Modèle chargé")
        return self.model