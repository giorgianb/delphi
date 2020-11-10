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


    # batch them so that they better resemble "real world" data - topics don't switch
    # from sentence to sentence
    sentence_sequence = list(chunks(sentence_sequence, batch_shuffle_size))
    topic_sequence = list(chunks(topic_sequence, batch_shuffle_size))
    sentence_texts = list(chunks(sentence_texts, batch_shuffle_size))

    train_sentences, test_sentences, train_topics, test_topics, train_text, test_text = \
        train_test_split(
            sentence_sequence,
            topic_sequence,
            sentence_texts,
            test_size=test_size,
            random_state=42
    )


    sentence_sequence = list(flatten(sentence_sequence))
    train_sentences = list(flatten(train_sentences))
    test_sentences = list(flatten(test_sentences))

    topic_sequence = list(flatten(topic_sequence))
    train_topics = list(flatten(train_topics))
    test_topics = list(flatten(test_topics))

    train_text = list(flatten(train_text))
    test_text = list(flatten(test_text))

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

    processor = embeddings.BloomFilter(5, 128)
    topics, dataset, train_dataset, test_dataset, train_text, test_text = make_dataset(data, processor)
