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
            gaussian_kernel_size = 11,
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
        self.gaussian_kernel_size = gaussian_kernel_size
        assert gaussian_kernel_size % 2 == 1

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

        self._ln_input = nn.LayerNorm(self._sentence_embedding_size)
        self._ln_forward = nn.LayerNorm(self._lstm_size)
        self._ln_backwards = nn.LayerNorm(self._lstm_size)
        self._ln_squeeze = nn.LayerNorm(2 * self._squeeze_layer_size)

        self._loss_fn = nn.CrossEntropyLoss()
        self._std = self._segmentation_smoothing_std

        if weights_file:
            self._model.load_weights(weights_file)

        padding = (self.gaussian_kernel_size - 1) / 2
        self.gaussian_filter = torch.nn.Conv2d(
                1, 
                1, 
                self.gaussian_kernel_size, 
                padding=padding, 
                bias=False
        )
        kern = Sector.gaussian_kernel(self.kernel_size, self.segmentation_smoothing_std)
        kern = kern.reshape((1, 1, kern.shape[0], kern.shape[1]))
        self.gaussian_filter.weight = torch.Parameter(torch.from_numpy(kern))

    def gaussian_kernel(size, std):
        t = np.linspace(-(size - 1) / 2, (size - 1)/2, size)
        x, y = np.meshgrid(t, t)
        kernel = np.exp(-0.5 * (x**2 + y**2)/std**2)

        return kernel/np.sum(kernel)

    # sentence_embeddings is (time_steps, features)
    def bemd_segment(self, sentence_embeddings):
        sentence_embeddings = tf.expand_dims(sentence_embeddings, 0)
        e_f = self._forward_lstm(sentence_embeddings)
        e_b = self._backwards_lstm(sentence_embeddings)
        e_f = tf.squeeze(e_f, 0).numpy()
        e_b = tf.squeeze(e_b, 0).numpy()

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

    @staticmethod
    def _batch_pca(e, dim):
        with torch.no_grad():
            mu = torch.mean(e, dim=0)
            c = e - mu # each column represents a variable
            cov = (torch.einsum('sbi,sbj->bij', c, c)/c.shape[0]).cpu().numpy()

        evals, evecs = np.linalg.eig(cov)
        ind = np.argsort(evals, axis=-1)[:, ::-1][:, :dim]
        # eigenvectors are columns
        with torch.no_grad():
            evecs = evecs[np.arange(evals.shape[0]).reshape(-1, 1), :, ind]
            evecs = torch.from_numpy(evecs).to('cuda:0')
            return torch.einsum('sbf,bnf->bsn', c, evecs)

    def forward(self, sentence_batch, segment=True):
        x = self._ln_input(sentence_batch)
        x_f = self._forward_lstm(x)[0]
        x_b = self._backwards_lstm(torch.flip(x, dims=(0,)))[0]
        x_b = torch.flip(x_b, dims=(0,))

        x_f = self._ln_forward(x_f)
        x_b = self._ln_backwards(x_b)
        x_f = self._squeeze_activation(self._squeeze_layer(x_f))
        x_b = self._squeeze_activation(self._squeeze_layer(x_b))
        x = self._ln_squeeze(x)
        topic_preds = self._topic_layer(x)
        if segment:
            e_f = Sector._batch_pca(x_f, self.segmentation_space_size)
            e_b = Sector._batch_pca(x_b, self.segmentation_space_size)
            with torch.no_grad():
                e_f = self.gaussian_filter(e_f).squeeze()
                e_b = self.gaussian_filter(e_b).squeeze()

                e_f_norms = torch.sqrt(torch.einsum('bsf,bsf->bs', e_f, e_f))
                e_b_norms = torch.sqrt(torch.einsum('bsf,bsf->bs', e_b, e_b))

                e_f_a = e_f[:, :, :-1]
                e_f_b = e_f[:, :, 1:]
                e_f_a_norms = e_f_norms[:, :-1]
                e_f_b_norms = e_f_norms[:, 1:]
                d_f = torch.einsum('bsf,bsf->bs', e_f_a, e_f_b)/(e_f_a_norms * e_f_b_norms)

                e_b_a = e_b[:, :, :-1]
                e_b_b = e_b[:, :, 1:]
                e_b_a_norms = e_b_norms[:, :-1]
                e_b_b_norms = e_b_norms[:, 1:]
                d_b = torch.einsum('bsf,bsf->bs', e_b_a, e_b_b)/(e_b_a_norms * e_b_b_norms)

                d = torch.sqrt(d_f * d_b)
                pos = torch.argmax(d, dim=1)

            return topic_preds, pos

        return topic_preds

    @staticmethod
    def train(model, inputs, validation_data, epochs=1000, weights_file=None):
        if not model._training_model_built:
            self._build_model(training=True, weights_file=weights_file)

        cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        checkpoint_dir = CHECKPOINT_DIR_TEMPLATE.format(cur_time)
        os.mkdir(checkpoint_dir)
        model = model.to('cuda:0')
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        with open(checkpoint_dir + 'model_params', 'w') as f:
            print("SECTOR("
            "\n\tlstm_size =", model._lstm_size,
            "\n\tsentence_embedding_size =", model._sentence_embedding_size,
            "\n\tdocument_size =", model._document_size,
            "\n\ttopic_embedding_size =", model._topic_embedding_size,
            "\n\tbatch_size =", model._batch_size,
            "\n\tsegmentation_space_size =", model._segmentation_space_size,
            "\n\tsegmentation_smoothing_std =", model._segmentation_smoothing_std,
            "\n\tmulti_topic =", model._multi_topic,
            "\n)",
            file=f)

            accuracies = []
            val_accuracies = []
            for epoch in range(epochs):
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
                    preds = preds.view(-1, model._topic_embedding_size)
                    target = topics_batch.view(-1)

                    loss = model._loss_fn(preds, target)
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


        print(f'Best Validation Accuracy: {max(val_accuracies)}')
        return model, accuracies, val_accuracies


