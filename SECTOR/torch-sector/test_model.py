import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import model
import dataset
import csv

BATCH_SIZE = 1         # 10 documents per batch
SENTENCE_EMBEDDING_SIZE = 4096
LSTM_SIZE = 384
BUFFER_SIZE = int(1e4)

dataset.initialize_dataset_bloom(5, SENTENCE_EMBEDDING_SIZE)

m = model.Sector(LSTM_SIZE, SENTENCE_EMBEDDING_SIZE, len(dataset.topics), squeeze_layer_size=196)
topic_dict = {v: k for k, v in dataset.topics.items()}

if __name__ == '__main__':
    wrong = []
    correct = []

    try:
        m, ac, vacs = model.Sector.train(m, dataset.train_dataset, dataset.test_dataset, epochs=100)
    except KeyboardInterrupt:
        pass

#    for test_batch, text_batch in zip(test, text):
#        documents_batch, subjects_batch = test_batch
#        subjects_p_batch = np.argmax(m._model.predict(documents_batch), axis=-1)
#        minibatches = zip(text_batch, subjects_batch, subjects_p_batch)
#        for text_minibatch, subjects_minibatch, subjects_p_minibatch in minibatches:
#            features = zip(text_minibatch, subjects_minibatch, subjects_p_minibatch)
#            for text, subject, subject_p in features:
#                print(f"'{text}' -> {topic_dict[subject_p]}")
#                if subject != subject_p:
#                    wrong.append((text, subject, subject_p))
#                else:
#                    correct.append((text, subject, subject_p))
#
#    with open('wrong.csv', 'w') as f:
#        fout = csv.writer(f)
#        fout.writerows(wrong)
#
#    with open('correct.csv', 'w') as f:
#        fout = csv.writer(f)
#        fout.writerows(correct)
#
