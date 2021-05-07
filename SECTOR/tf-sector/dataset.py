import csv
import stanza
import tensorflow as tf
import embeddings
from sklearn.model_selection import train_test_split
import sys
import json
import random

csv.field_size_limit(sys.maxsize)

DATA_FILE = "micro-data.csv"
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

def group_by_topic(lst, topics):
    prev_topic = topics[0]
    cur_group = [lst[0]]
    groups = []

    for item, topic in zip(lst[1:], topics[1:]):
        if topic == prev_topic:
            cur_group.append(item)
        else:
            groups.append(cur_group)
            cur_group = [item]
            prev_topic = topic

    groups.append(cur_group)
    return groups


# assumes data is tokenized and lemmatized
def make_dataset(data, processor, test_size=0.2, random_state=42):
    sentence_sequence = []
    sentence_texts = []
    topic_sequence = []

    topic_counter = 0
    topics = {}
    for topic, parsed, sentences in data:
        if topic not in topics:
            topics[topic] = topic_counter
            topic_counter += 1

        for sentence, text in zip(parsed, sentences):
            sentence_sequence.append(processor(sentence))
            sentence_texts.append(' '.join(text))
            topic_sequence.append(topics[topic])

    sentence_groups = group_by_topic(sentence_sequence, topic_sequence)
    sentence_text_groups = group_by_topic(sentence_texts, topic_sequence)
    topic_groups = group_by_topic(topic_sequence, topic_sequence)

    train_sentence_groups = []
    train_sentence_text_groups = []
    train_topic_groups = []
    test_sentences = []
    test_sentence_texts = []
    test_topics = []
    random.seed(random_state)
    for s_group, st_group, t_group in zip(sentence_groups, sentence_text_groups, topic_groups):
        n = len(s_group)
        assert len(s_group) == len(st_group)
        assert len(st_group) == len(t_group)
        tr_indices, te_indices = train_test_split(
                tuple(range(n)), 
                test_size=test_size, 
                random_state=random.randint(0, int(2**32))
        )

        train_sentences = [s_group[i] for i in tr_indices]
        train_sentence_texts = [st_group[i] for i in tr_indices]
        train_topics = [t_group[i] for i in tr_indices]
        train_sentence_groups.append(train_sentences)
        train_sentence_text_groups.append(train_sentence_texts)
        train_topic_groups.append(train_topics)
        test_sentences.extend(s_group[i] for i in te_indices)
        test_topics.extend(t_group[i] for i in te_indices)
        test_sentence_texts.extend(st_group[i] for i in te_indices)

    # Generate the dataset
    def generate_dataset():
        new_topic_order = list(range(len(train_sentence_groups)))
        random.shuffle(new_topic_order)
        tsg = train_sentence_groups
        tstg = train_sentence_text_groups
        ttg = train_topic_groups
        for i in new_topic_order:
            new_sentence_order = list(range(len(tsg[i])))
            random.shuffle(new_sentence_order)
            for j in new_sentence_order:
                yield tsg[i][j], tstg[i][j], ttg[i][j]

    EMBEDDING_DIM = len(sentence_sequence[0])
    train_dataset = tf.data.Dataset.from_generator(
            generate_dataset,
            output_signature=(
                tf.TensorSpec(shape=(EMBEDDING_DIM,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
    )

    sentence_sequence = tf.data.Dataset.from_tensor_slices(sentence_sequence)
    topic_sequence = tf.data.Dataset.from_tensor_slices(topic_sequence)
    sentence_texts = tf.data.Dataset.from_tensor_slices(sentence_texts)
    dataset = tf.data.Dataset.zip((sentence_sequence, sentence_texts, topic_sequence))

    test_sentences = tf.data.Dataset.from_tensor_slices(test_sentences)
    test_topics = tf.data.Dataset.from_tensor_slices(test_topics)
    test_sentence_texts = tf.data.Dataset.from_tensor_slices(test_sentence_texts)
    test_dataset = tf.data.Dataset.zip((
        test_sentences, 
        test_sentence_texts, 
        test_topics
        ))

    return topics, dataset, train_dataset, test_dataset

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

def initialize_dataset_bloom(n_hash_functions, sentence_embedding_size, random_state=42):
    global topics, dataset, train_dataset, test_dataset
    processor = embeddings.BloomFilter(n_hash_functions, sentence_embedding_size)
    datasets = make_dataset(data, processor, random_state=random_state)
    topics, dataset, train_dataset, test_dataset = datasets
