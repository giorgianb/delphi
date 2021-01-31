import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import model
import dataset
import csv

BATCH_SIZE = 1         # 10 documents per batch
DOCUMENT_SIZE = 1200     # 100 sentences per document
SENTENCE_EMBEDDING_SIZE = 4096
LSTM_SIZE = 256
BUFFER_SIZE = int(1e4)

dataset.initialize_dataset_bloom(5, SENTENCE_EMBEDDING_SIZE)

m = model.Sector(
        LSTM_SIZE,
        SENTENCE_EMBEDDING_SIZE,
        DOCUMENT_SIZE,
        len(dataset.topics),
        BATCH_SIZE
)

train = dataset.train_dataset.batch(DOCUMENT_SIZE, drop_remainder=True)
train = train.batch(BATCH_SIZE, drop_remainder=True)

test = dataset.test_dataset.batch(DOCUMENT_SIZE, drop_remainder=True)
test = test.batch(BATCH_SIZE, drop_remainder=True)
text = list(dataset.chunks(dataset.test_text, DOCUMENT_SIZE))
text = list(dataset.chunks(text, BATCH_SIZE))
all_text = list(dataset.chunks(dataset.test_text, BATCH_SIZE))
topic_dict = {v: k for k, v in dataset.topics.items()}

wrong = []
correct = []

m.train(train, validation_data=test, epochs=100)
for test_batch, text_batch in zip(test, text):
    documents_batch, subjects_batch = test_batch
    subjects_p_batch = np.argmax(m._model.predict(documents_batch), axis=-1)
    minibatches = zip(text_batch, subjects_batch, subjects_p_batch)
    for text_minibatch, subjects_minibatch, subjects_p_minibatch in minibatches:
        features = zip(text_minibatch, subjects_minibatch, subjects_p_minibatch)
        for text, subject, subject_p in features:
            print(f"'{text}' -> {topic_dict[subject_p]}")
            if subject != subject_p:
                wrong.append((text, subject, subject_p))
            else:
                correct.append((text, subject, subject_p))

with open('wrong.csv', 'w') as f:
    fout = csv.writer(f)
    fout.writerows(wrong)

with open('correct.csv', 'w') as f:
    fout = csv.writer(f)
    fout.writerows(correct)

