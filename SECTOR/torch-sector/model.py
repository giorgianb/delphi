import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import datetime
import os
import gc

CHECKPOINT_DIR_TEMPLATE = 'sector_training/{}/'
CHECKPOINT_NAME = 'sector.ckpt'

class Sector(nn.Module):
    def __init__(
            self, 
            lstm_size, 
            sentence_embedding_size,
            document_size,
            topic_embedding_size, 
            batch_size,
            squeeze_layer_size = 64,
            segmentation_space_size = 16,
            segmentation_smoothing_std = 2.5,
            multi_topic=False
            ):
        super(Sector, self).__init__()
        self._lstm_size = lstm_size
        self._document_size = document_size
        self._sentence_embedding_size = sentence_embedding_size
        self._topic_embedding_size = topic_embedding_size
        self._batch_size = batch_size
        self._segmentation_space_size = segmentation_space_size
        self._segmentation_smoothing_std = segmentation_smoothing_std
        self._multi_topic = multi_topic
        self._training_model_built = True
        self._model = None
        self._squeeze_layer_size = squeeze_layer_size

        self._build_model()

    def _build_model(self, training=True, weights_file=None):
        batch_size = self._batch_size if training else 1
        document_size = self._document_size if training else None
        embedding_size = self._sentence_embedding_size
        self._training_model_built = training
        self._forward_lstm = nn.LSTM(
                input_size=self._sentence_embedding_size,
                hidden_size=self._lstm_size,
        )
        self._backwards_lstm = nn.LSTM(
                input_size=self._sentence_embedding_size,
                hidden_size=self._lstm_size,
        )

        self._squeeze_layer = nn.Linear(
                self._lstm_size,
                self._squeeze_layer_size
        )

        self._topic_layer = nn.Linear(
                2 * self._squeeze_layer_size,
                self._topic_embedding_size
        )

        self._squeeze_activation = torch.nn.Tanh()
        self._topic_activation = torch.nn.Softmax(dim=-1)

        self._ln_input = nn.LayerNorm(self._sentence_embedding_size)
        self._ln_forward = nn.LayerNorm(self._lstm_size)
        self._ln_backwards = nn.LayerNorm(self._lstm_size)
        self._ln_squeeze = nn.LayerNorm(2 * self._squeeze_layer_size)

        self._loss_fn = nn.CrossEntropyLoss()
        self._pca = PCA(n_components = self._segmentation_space_size)
        self._std = self._segmentation_smoothing_std

        if weights_file:
            self._model.load_weights(weights_file)

    def forward(self, sentence_batch):
        x = self._ln_input(sentence_batch)
        x_f = self._forward_lstm(x)[0]
        x_b = self._backwards_lstm(torch.flip(x, dims=(0,)))[0]
        x_b = torch.flip(x_b, dims=(0,))

        x_f = self._ln_forward(x_f)
        x_b = self._ln_backwards(x_b)
        x_f = self._squeeze_activation(self._squeeze_layer(x_f))
        x_b = self._squeeze_activation(self._squeeze_layer(x_b))
        x = torch.cat((x_f, x_b), dim=-1)
        x = self._ln_squeeze(x)

        return self._topic_layer(x)

    def train(self, inputs, validation_data, epochs=1000, weights_file=None):
        if not self._training_model_built:
            self._build_model(training=True, weights_file=weights_file)

        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        checkpoint_dir = CHECKPOINT_DIR_TEMPLATE.format(cur_time)
        os.mkdir(checkpoint_dir)
        model = self.to('cuda:0')
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

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

            accuracies = []
            val_accuracies = []
            for epoch in range(epochs):
                if epoch > 10:
                    print("Adjusting learning rate to 0.005")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.005
                if epoch > 20:
                    print("Adjusting learning rate to 0.001")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.001
                if epoch > 40:
                    for param_group in optim.param_groups:
                        print("Adjusting learning rate to 0.0005")
                        param_group['lr'] = 0.0005
                if epoch > 80:
                    print("Adjusting learning rate to 0.00001")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.00001
                if epoch > 160:
                    print("Adjusting learning rate to 0.01")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.01
                if epoch > 170:
                    print("Adjusting learning rate to 0.0005")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.005
                if epoch > 190:
                    print("Adjusting learning rate to 0.0001")
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.001
                if epoch > 230:
                    for param_group in optim.param_groups:
                        param_group['lr'] = 0.0001


                print(f'Epoch {epoch + 1}/{epochs}')
                losses = []
                inputs.sampler.generate_document_size()
                length = len(inputs)
                batches = tuple(inputs)
                for i, (sentence_batch, text_batch, topics_batch) in enumerate(batches):
                    optim.zero_grad()
                    sentence_batch = sentence_batch.to('cuda')
                    topics_batch = topics_batch.to('cuda')
                    preds = model(sentence_batch)
                    preds = preds.view(-1, self._topic_embedding_size)
                    target = topics_batch.view(-1)

                    loss = self._loss_fn(preds, target)
                    losses.append(loss)

                    print(f'batch {i+1}/{length}: loss: {loss:0.8f}', end='\r')
                    loss.backward()
                    optim.step()

                print(f'average loss: {sum(losses)/len(losses)}')
                with torch.no_grad():
                    total_correct = 0
                    total = 0
                    for sentence_batch, text_batch, topics_batch in batches:
                        sentence_batch = sentence_batch.to('cuda')
                        topics_batch = topics_batch.to('cuda')
                        preds = torch.argmax(model(sentence_batch), dim=-1)
                        total_correct += torch.sum(preds == topics_batch).cpu().numpy()
                        total += sentence_batch.shape[0] * sentence_batch.shape[1]

                print(f'training accuracy: {total_correct/total}')
                accuracies.append(total_correct/total)
                with torch.no_grad():
                    total_correct = 0
                    total = 0
                    validation_data.sampler.generate_document_size()
                    for sentence_batch, text_batch, topics_batch in validation_data:
                        sentence_batch = sentence_batch.to('cuda')
                        topics_batch = topics_batch.to('cuda')
                        preds = torch.argmax(model(sentence_batch), dim=-1)
                        total_correct += torch.sum(preds == topics_batch).cpu().numpy()
                        total += sentence_batch.shape[0] * sentence_batch.shape[1]

                print(f'validation accuracy: {total_correct/total}')
                val_accuracies.append(total_correct/total)


        print('Best Validation Accuracy: {max(val_accuracies)}')
        return model

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
