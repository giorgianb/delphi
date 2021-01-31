import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import datetime
import os

CHECKPOINT_DIR_TEMPLATE = 'sector_training/{}/'
CHECKPOINT_NAME = 'sector.ckpt'

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
        self._lstm_size = lstm_size
        self._sentence_embedding_size = sentence_embedding_size
        self._document_size = document_size
        self._topic_embedding_size = topic_embedding_size
        self._batch_size = batch_size
        self._segmentation_space_size = segmentation_space_size
        self._segmentation_smoothing_std = segmentation_smoothing_std
        self._multi_topic = multi_topic
        self._training_model_built = True
        self._model = None

        self._build_model()

    def _build_model(self, training=True, weights_file=None):
        if not weights_file and self._model is not None:
            weights_file = 'model.ckpt'
            self._model.save_weights(weights_file)

        batch_size = self._batch_size if training else 1
        document_size = self._document_size if training else None
        self._training_model_built = training
        self._forward_lstm = tf.keras.layers.LSTM(
                self._lstm_size, 
                recurrent_initializer='glorot_uniform',
                return_sequences=True,
                stateful=training
        )
        self._backwards_lstm = tf.keras.layers.LSTM(
                self._lstm_size, 
                recurrent_initializer='glorot_uniform',
                return_sequences=True,
                stateful=training,
                go_backwards=True
        )

        self._bi_lstm = tf.keras.layers.Bidirectional(
                self._forward_lstm,
                backward_layer=self._backwards_lstm,
                batch_input_shape=[
                    batch_size, 
                    document_size,
                    self._sentence_embedding_size
                    ],
                merge_mode='sum'
        )

        act = 'sigmoid' if self._multi_topic else 'softmax'
        self._topic_layer = tf.keras.layers.Dense(self._topic_embedding_size, activation=act)

        self._model = tf.keras.models.Sequential((
            tf.keras.layers.LayerNormalization(dtype=float),
            self._bi_lstm,
            tf.keras.layers.LayerNormalization(dtype=float),
            self._topic_layer
        ))

        spc = tf.keras.losses.sparse_categorical_crossentropy
        loss = lambda labels, logits: spc(labels, logits, from_logits=True)
        self._model.compile(
                optimizer='adam',
                loss=loss,
                metrics=['accuracy']
        )

        self._pca = PCA(n_components = self._segmentation_space_size)
        self._std = self._segmentation_smoothing_std

        if weights_file:
            self._model.load_weights(weights_file)

    def train(self, inputs, epochs=1000, validation_data=None, weights_file=None):
        if not self._training_model_built:
            self._build_model(training=True, weights_file=weights_file)

        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        checkpoint_dir = CHECKPOINT_DIR_TEMPLATE.format(cur_time)
        os.mkdir(checkpoint_dir)

        with open(checkpoint_dir + 'model_params', 'w') as f:
            print("SECTOR("
            "\n\tlstm_size =", self._lstm_size,
            "\n\tsentence_embedding_size =", self._sentence_embedding_size,
            "\n\tdocument_size =", self._document_size,
            "\n\ttopic_embedding_size =", self._topic_embedding_size,
            "\n\tbatch_size =", self._batch_size,
            "\n\tsegmentation_space_size =", self._segmentation_space_size,
            "\n\tsegmentation_smoothing_std =", self._segmentation_smoothing_std,
            "\n\tmulti_topic =", self._multi_topic,
            "\n)",
            file=f)

        checkpoint_path = checkpoint_dir + CHECKPOINT_NAME

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        if validation_data is None:
            history = self._model.fit(
                    inputs, 
                    epochs=epochs, 
                    callbacks=[cp_callback]
            )
        else:
            es_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    mode='max',
                    min_delta=0.0,
                    patience=15,
                    verbose=True,
                    restore_best_weights=True
            )
            history = self._model.fit(
                    inputs, 
                    epochs=epochs, 
                    validation_data=validation_data, 
                    callbacks=[cp_callback, es_callback]
            )


        with open(checkpoint_dir + 'history', 'w') as f:
            print(history.history, file=f)


        return history

    def predict(self, sentences, weights_file=None):
        if self._training_model_built:
            self._build_model(training=False, weights_file=weights_file)

        sentences = tf.expand_dims(sentences, 0)
        predicted_topics = self._model(sentences)
        predicted_topics = tf.squeeze(predicted_topics, 0)

        return predicted_topics
        
    # sentence_embeddings is (time_steps, features)
    def bemd_segment(self, sentence_embeddings):
        sentence_embeddings = tf.expand_dims(sentence_embeddings, 0)
        e_f = self._forward_lstm(sentence_embeddings)
        e_b = self._backwards_lstm(sentence_embeddings)
        e_f = tf.squeeze(e_f, 0).numpy()
        e_b = tf.squeeze(e_b, 0).numpy()

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
