import csv
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import Tensor
import torch
import embeddings
from sklearn.model_selection import train_test_split
import sys
import json
import random
import itertools
import numpy as np
import math

csv.field_size_limit(sys.maxsize)
class TopicDataset(Dataset):
    def __init__(self, grouped_sentences,  grouped_sentence_texts, grouped_topics):
        super(TopicDataset, self).__init__()
        self._sentences = tuple(tuple(group) for group in grouped_sentences)
        self._sentence_texts = tuple(tuple(group) for group in grouped_sentence_texts)
        self._topics = tuple(tuple(group) for group in grouped_topics)
        self._group_lengths = tuple(len(group) for group in grouped_sentences)

    @property
    def group_lengths(self):
        return self._group_lengths

    def __getitem__(self, indices):
        sents = self._sentences
        texts = self._sentence_texts
        topics = self._topics
        sents_minibatch = Tensor([sents[gi][wgi] for gi, wgi in indices])
        texts_minibatch = [texts[gi][wgi] for gi, wgi in indices]
        topics_minibatch = Tensor([topics[gi][wgi] for gi, wgi in indices])

        return (sents_minibatch, texts_minibatch, topics_minibatch)

    def __len__(self):
        return len(self._key_lengths)

class TopicBatchSampler(Sampler):
    def __init__(self, group_lengths, min_document_size, max_document_size, train=True):
        self.train = train
        self.min_document_size = min_document_size
        self.max_document_size = max_document_size
        self._indices = tuple(tuple(range(group_length)) for group_length in group_lengths)
        self._total = sum(group_lengths)

    def __iter__(self):
        group_order = np.arange(len(self._indices))
        ds = random.randint(self.min_document_size, self.max_document_size + 1)
        perm = lambda x: np.random.permutation(x) if self.train else x
        indices = self._indices
        if self.train:
            group_order = np.random.permutation(group_order)

        fi = tuple(
                (g_i, w_g_i) for g_i in group_order for w_g_i in perm(indices[g_i])
        )
        documents = filter(
                lambda document: len(document) == ds, 
                (fi[i:i + ds] for i in range(0, len(fi), ds))
        )
        return documents


DATA_FILE = "micro-data.csv"
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
def make_dataset(data, processor, test_size=0.2, batch_size=8,random_state=42):
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
    test_sentence_groups = []
    test_sentence_text_groups = []
    test_topic_groups = []
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
        test_sentences = [s_group[i] for i in te_indices]
        test_sentence_texts = [st_group[i] for i in te_indices]
        test_topics = [t_group[i] for i in te_indices]

        train_sentence_groups.append(train_sentences)
        train_sentence_text_groups.append(train_sentence_texts)
        train_topic_groups.append(train_topics)
        test_sentence_groups.append(test_sentences)
        test_sentence_text_groups.append(test_sentence_texts)
        test_topic_groups.append(test_topics)

    dataset = TopicDataset(
            sentence_groups, 
            sentence_text_groups, 
            topic_groups
    )
    batch_sampler = TopicBatchSampler(
            dataset.group_lengths, 
            32,
            512,
    )

    train_dataset = TopicDataset(
            train_sentence_groups, 
            train_sentence_text_groups, 
            train_topic_groups
    )
    train_batch_sampler = TopicBatchSampler(
            train_dataset.group_lengths, 
            32,
            512,
    )

    test_dataset = TopicDataset(
            test_sentence_groups, 
            test_sentence_text_groups, 
            test_topic_groups
    )
    test_batch_sampler = TopicBatchSampler(
            test_dataset.group_lengths, 
            32,
            512,
            train=False
    )

    def smart_collate(batch):
        sentences = torch.stack(tuple(mb[0] for mb in batch))
        texts = tuple(mb[1] for mb in batch)
        topics = torch.stack(tuple(mb[2] for mb in batch))
        return sentences, texts, topics

    train_dataloader = DataLoader(
            train_dataset, 
            sampler=train_batch_sampler, 
            batch_size=batch_size,
            collate_fn=smart_collate
    )
    test_dataloader = DataLoader(
            test_dataset, 
            sampler=test_batch_sampler, 
            batch_size=batch_size,
            collate_fn=smart_collate
    )

    dataloader = DataLoader(
            test_dataset, 
            sampler=test_batch_sampler, 
            batch_size=batch_size,
            collate_fn=smart_collate
    )

    return topics, dataloader, train_dataloader, test_dataloader

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
