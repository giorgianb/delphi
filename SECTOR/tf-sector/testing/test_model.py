import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import model
import dataset

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

m.train(train, validation_data=test, epochs=100)
for test_batch, text_batch in zip(test, text):
    test_document = test_batch[0]
    preds = m._model.predict(test_document)
    print(preds.shape)
    print(test_document.shape)
    print(len(text_batch))
    print(len(text_batch[0]))
    for text_doc, pred_doc in zip(text_batch, preds):
        for text, pred in zip(text_doc, pred_doc):
            print("{} -> {}".format(' '.join(text), topic_dict[np.argmax(pred)]))
