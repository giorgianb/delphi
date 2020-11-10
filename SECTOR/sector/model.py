import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import os.path

CHECKPOINT_PATH = 'sector_training/sector.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

class Sector:
    def __init__(
            self, 
            lstm_size, 
            sentence_embedding_size,
            document_size,
            topic_embedding_size, 
            batch_size,
            segmentation_space_size = 16,
            segmentation_smoothing_std = 2.5,
            multi_topic=False
            ):
        self._forward_lstm = tf.keras.layers.LSTM(
                lstm_size, 
                recurrent_initializer='glorot_uniform',
                return_sequences=True,
                stateful=True
        )
        self._backwards_lstm = tf.keras.layers.LSTM(
                lstm_size, 
                recurrent_initializer='glorot_uniform',
                return_sequences=True,
                stateful=True,
                go_backwards=True
        )

        self._bi_lstm = tf.keras.layers.Bidirectional(
                self._forward_lstm,
                backward_layer=self._backwards_lstm,
                batch_input_shape=[batch_size, document_size, sentence_embedding_size],
                merge_mode='sum'
        )

        act = 'sigmoid' if multi_topic else 'softmax'
        self._topic_layer = tf.keras.layers.Dense(topic_embedding_size, activation=act)

        self._model = tf.keras.models.Sequential((
                self._bi_lstm,
                self._topic_layer
        ))

        spc = tf.keras.losses.sparse_categorical_crossentropy
        loss = lambda labels, logits: spc(labels, logits, from_logits=True)
        self._model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy']
        )

        self._pca = PCA(n_components = segmentation_space_size)
        self._std = segmentation_smoothing_std

    def train(self, inputs, validation_data=None):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
        if validation_data is None:
            self._model.fit(inputs, epochs=1000, callbacks=[cp_callback])
        else:
            self._model.fit(
                    inputs, 
                    epochs=1000, 
                    validation_data=validation_data, 
                    callbacks=[cp_callback]
            )

    def predict(self, inputs):
        pass

    # sentence_embeddings is (time_steps, features)
    def bemd_segment(self, sentence_embeddings):
        e_f = self._forward_lstm(sentence_embeddings).numpy()
        e_b = self._backwards_lstm(sentence_embeddings).numpy()

        e_f = self._pca.fit_transform(e_f)
        e_b = self._pca.fit_transform(e_b)

        e_f = gaussian_filter(e_f, self._std)
        e_b = gaussian_filter(e_b, self._std)

        d_f = np.sum(e_f[1:]*e_f[:-1], axis=1)
        df /= (np.linalg.norm(e_f[1:], axis=1) * np.linalg.norm(e_f[:-1], axis=1))

        d_b = np.sum(e_b[1:] * e_b[:-1])
        d_b /= (np.linalg.norm(e_b[1:], axis=1) * np.linalg.norm(e_b[:-1], axis=1))

        # take the geometric mean
        d = np.sqrt(d_f * d_b)

        segment_edges, _ = find_peaks(d)

        return segment_edges
