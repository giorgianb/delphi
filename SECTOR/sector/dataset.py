import csv
import stanza
import tensorflow as tf
import embeddings
from sklearn.model_selection import train_test_split
import sys
import json
import random

csv.field_size_limit(sys.maxsize)

DATA_FILE = "data.csv"

# shuffle the order of the sentences within each topic group
# also shuffle the order of the topics
@tf.function
def within_topic_shuffle(batch):
    print(type(batch))
    print(batch.shape)
    sentences = tf.unstack(sentences)
    topics = tf.unstack(topics)
    s_groups = []
    t_groups = []
    prev_topic = topics[0]
    current_s_group = [sentences[0]]
    current_t_group = [topics[0]]
    for s, t in zip(sentences[1:], topics[1:]):
        if t.numpy() == prev_topic.numpy():
            current_s_group.append(s)
            current_t_group.append(t)
        else:
            s_groups.append(current_s_group)
            t_groups.append(current_t_group)
            prev_topic = t
            current_s_group = [s]
            current_t_group = [t]
    s_groups.append(current_s_group)
    t_groups.append(current_t_group)

    for s_group in s_groups:
        random.shuffle(s_group)

    s = random.getstate()
    random.setstate(s)
    random.shuffle(s_groups)
    random.setstate(s)
    random.shuffle(t_groups)

    new_sentences = [s for group in s_groups for s in group]
    new_topics = [t for group in t_groups for t in group]

    return tf.stack(new_sentences), tf.stack(new_topics)

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
def make_dataset(data, processor, test_size=0.2, batch_sample_size=100, random_state=42):
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

    sentence_sequence = list(chunks(sentence_sequence, batch_sample_size))
    topic_sequence = list(chunks(topic_sequence, batch_sample_size))
    sentence_texts = list(chunks(sentence_texts, batch_sample_size))
    for ss, ts, st in zip(sentence_sequence, topic_sequence, sentence_texts):
        n = len(ss)
        tr_indices, te_indices = train_test_split(
                tuple(range(n)), 
                test_size=test_size, 
                random_state=random_state
                )
        tr_indices = sorted(tr_indices)
        te_indices = sorted(te_indices)

        train_sentences.extend(ss[i] for i in tr_indices)
        test_sentences.extend(ss[i] for i in te_indices)
        train_topics.extend(ts[i] for i in tr_indices)
        test_topics.extend(ts[i] for i in te_indices)
        train_text.extend(st[i] for i in tr_indices)
        test_text.extend(st[i] for i in te_indices)

    sentence_sequence = list(flatten(sentence_sequence))
    topic_sequence = list(flatten(topic_sequence))
    sentence_texts = list(flatten(sentence_texts))


    # Generate the dataset
    def generate_dataset():

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

def initialize_dataset_bloom(n_hash_functions, sentence_embedding_size, random_state=42):
    global topics, dataset, train_dataset, test_dataset, train_text, test_text
    processor = embeddings.BloomFilter(n_hash_functions, sentence_embedding_size)
    datasets = make_dataset(data, processor, random_state=random_state)
    topics, dataset, train_dataset, test_dataset, train_text, test_text = datasets
