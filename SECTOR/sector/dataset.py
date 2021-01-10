import csv
import stanza
import tensorflow as tf
import embeddings
from sklearn.model_selection import train_test_split
import sys
import json

csv.field_size_limit(sys.maxsize)

DATA_FILE = "data.csv"
def tokenize_and_lemmatize_data(data):
    nlp = stanza.Pipeline('en', processors='tokenize,lemma')
    rows = []

    for topic, text in data:
        doc = nlp(text)
        sentences = [[word.lemma for word in sent.words] for sent in doc.sentences]
        rows.append((topic, sentences))

    return rows

def chunks(lst, n):
    return (lst[i:i + n] for i in range(0, len(lst), n))

def flatten(lst):
    return (item for sublist in lst for item in sublist)

# assumes data is tokenized and lemmatized
def make_dataset(data, processor, test_size=0.2, batch_shuffle_size=100):
    sentence_sequence = []
    topic_sequence = []
    sentence_texts = []

    topic_counter = 0
    topics = {}
    for topic, parsed, sentences in data:
        if topic not in topics:
            topics[topic] = topic_counter
            topic_counter += 1

        for sentence, text in zip(parsed, sentences):
            sentence_sequence.append(processor(sentence))
            sentence_texts.append(text)
            topic_sequence.append(topics[topic])

    train_sentences = []
    test_sentences = []
    train_topics = []
    test_topics = []
    train_text = []
    test_text = []

    sentence_sequence = list(chunks(sentence_sequence, batch_shuffle_size))
    topic_sequence = list(chunks(topic_sequence, batch_shuffle_size))
    sentence_texts = list(chunks(sentence_texts, batch_shuffle_size))
    for ss, ts, st in zip(sentence_sequence, topic_sequence, sentence_texts):
        if len(ss) < batch_shuffle_size:
            continue

        tr_sentences, te_sentences, tr_topics, te_topics, tr_text, te_text = \
                train_test_split(ss, ts, st, test_size=test_size, random_state=42)

        train_sentences.extend(tr_sentences)
        test_sentences.extend(te_sentences)
        train_topics.extend(tr_topics)
        test_topics.extend(te_topics)
        train_text.extend(tr_text)
        test_text.extend(te_text)

    sentence_sequence = list(flatten(sentence_sequence))
    topic_sequence = list(flatten(topic_sequence))
    sentence_texts = list(flatten(sentence_texts))

    print("TOTAL SENTENCES: {}".format(len(sentence_sequence)))
    sentence_sequence = tf.data.Dataset.from_tensor_slices(sentence_sequence)
    topic_sequence = tf.data.Dataset.from_tensor_slices(topic_sequence)
    train_sentences = tf.data.Dataset.from_tensor_slices(train_sentences)
    train_topics = tf.data.Dataset.from_tensor_slices(train_topics)
    test_sentences = tf.data.Dataset.from_tensor_slices(test_sentences)
    test_topics = tf.data.Dataset.from_tensor_slices(test_topics)

    dataset = tf.data.Dataset.zip((sentence_sequence, topic_sequence))
    train_dataset = tf.data.Dataset.zip((train_sentences, train_topics))
    test_dataset = tf.data.Dataset.zip((test_sentences, test_topics))


    return topics, dataset, train_dataset, test_dataset, train_text, test_text

with open(DATA_FILE) as f:
    data = list(csv.reader(f))
    titles = (row[0] for row in data)
    parsed = (json.loads(row[1]) for row in data)
    text = (json.loads(row[2]) for row in data)
    data = tuple(zip(titles, parsed, text))
    topics = None
    dataset = None 
    train_dataset = None
    test_dataset = None
    train_text = None
    test_text = None

def initialize_dataset_bloom(n_hash_functions, sentence_embedding_size):
    global topics, dataset, train_dataset, test_dataset, train_text, test_text
    processor = embeddings.BloomFilter(n_hash_functions, sentence_embedding_size)
    datasets = make_dataset(data, processor)
    topics, dataset, train_dataset, test_dataset, train_text, test_text = datasets
