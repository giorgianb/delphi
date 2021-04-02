import csv
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import tensor, Tensor
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
        texts_minibatch = np.array([texts[gi][wgi] for gi, wgi in indices])
        # use tensor to keep tensor as integer
        topics_minibatch = tensor([topics[gi][wgi] for gi, wgi in indices])

        return (sents_minibatch,
                texts_minibatch, 
                topics_minibatch)

    def __len__(self):
        return len(self._key_lengths)

class TopicBatchSampler(Sampler):
    def __init__(self, group_lengths, min_document_size, max_document_size, train=True):
        self.train = train
        self.min_document_size = min_document_size
        self.max_document_size = max_document_size
        self._indices = tuple(tuple(range(group_length)) for group_length in group_lengths)
        self._total = sum(group_lengths)

    def generate_document_size(self):
        self._document_size = random.randint(self.min_document_size, self.max_document_size + 1)

    def __len__(self):
        return int(math.floor(self._total / self._document_size))

    def __iter__(self):
        group_order = np.arange(len(self._indices))
        perm = lambda x: np.random.permutation(x) if self.train else x
        ds = self._document_size
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
def make_dataset(data, processor, test_size=0.2, batch_size=8, random_state=42, allowed_topic={}):
    sentence_sequence = []
    sentence_texts = []
    topic_sequence = []

    topic_counter = 0
    topics = {}
    for topic, parsed, sentences in data:
        if topic not in allowed_topic:
            continue

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
            1,
            512,
    )

    train_dataset = TopicDataset(
            train_sentence_groups, 
            train_sentence_text_groups, 
            train_topic_groups
    )
    train_batch_sampler = TopicBatchSampler(
            train_dataset.group_lengths, 
            1,
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
        sentences = torch.stack(tuple(mb[0] for mb in batch), dim=1)
        texts = np.stack(tuple(mb[1] for mb in batch), axis=1)
        topics = torch.stack(tuple(mb[2] for mb in batch), dim=1)
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


DATA_FILE = "micro-data.csv"


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
    global topics, dataset, train_dataset, test_dataset, DATA_TOPICS
    processor = embeddings.BloomFilter(n_hash_functions, sentence_embedding_size)
    datasets = make_dataset(data, processor, random_state=random_state, DATA_TOPICS)
    topics, dataset, train_dataset, test_dataset = datasets


DATA_TOPICS = {
    "PROLOGUE I: PROLOGUE TO ELECTRONICS",
    "Brief History",
    "Passive and Active Devices",
    "Electronic Circuits",
    "Discrete and Integrated Circuits",
    "Analog and Digital Signals",
    "Notation",
    "Summary",
    "PART 1: SEMICONDUCTOR DEVICES AND BASIC APPLICATIONS",
    "Chapter 1: Semiconductor Materials and Diodes",
    "Preview",
    "1.1 Semiconductor Materials and Properties",
    "1.2 The pn Junction",
    "1.3 Diode Circuits: DC Analysis and Models",
    "1.4 Diode Circuits: AC Equivalent Circuit",
    "1.5 Other Diode Types",
    "1.6 Design Application: Diode Thermometer",
    "1.7 Summary",
    "Chapter 2: Diode Circuits",
    "2.1 Rectifier Circuits",
    "2.2 Zener Diode Circuits",
    "2.3 Clipper and Clamper Circuits",
    "2.4 Multiple-Diode Circuits",
    "2.5 Photodiode and LED Circuits",
    "2.6 Design Application: DC Power Supply",
    "2.7 Summary",
    "Chapter 3: The Field-Effect Transistor",
    "3.1 MOS Field-Effect Transistor",
    "3.2 MOSFET DC Circuit Analysis",
    "3.3 Basic MOSFET Applications: Switch, Digital Logic Gate, and Amplifier",
    "3.4 Constant-Current Biasing",
    "3.5 Multistage MOSFET Circuits",
    "3.6 Junction Field-Effect Transistor",
    "3.7 Design Application: Diode Thermometer with an MOS Transistor",
    "3.8 Summary",
    "Chapter 4: Basic FET Amplifiers",
    "4.1 The MOSFET Amplifier",
    "4.2 Basic Transistor Amplifier Configurations",
    "4.3 The Common-Source Amplifier",
    "4.4 The Common-Drain (Source-Follower) Amplifier",
    "4.5 The Common-Gate Configuration",
    "4.6 The Three Basic Amplifier Configurations: Summary and Comparison",
    "4.7 Single-Stage Integrated Circuit MOSFET Amplifiers",
    "4.8 Multistage Amplifiers",
    "4.9 Basic JFET Amplifiers",
    "4.10 Design Application: A Two-Stage Amplifier",
    "4.11 Summary",
    "Chapter 5: The Bipolar Junction Transistor",
    "5.1 Basic Bipolar Junction Transistor",
    "5.2 DC Analysis of Transistor Circuits",
    "5.3 Basic Transistor Applications",
    "5.4 Bipolar Transistor Biasing",
    "5.5 Multistage Circuits",
    "5.6 Design Application: Diode Thermometer with a Bipolar Transistor",
    "5.7 Summary",
    "Chapter 6: Basic BJT Amplifiers",
    "6.1 Analog Signals and Linear Amplifiers",
    "6.2 The Bipolar Linear Amplifier",
    "6.3 Basic Transistor Amplifier Configurations",
    "6.4 Common-Emitter Amplifiers",
    "6.5 AC Load Line Analysis",
    "6.6 Common-Collector (Emitter-Follower) Amplifier",
    "6.7 Common-Base Amplifier",
    "6.8 The Three Basic Amplifiers: Summary and Comparison",
    "6.9 Multistage Amplifiers",
    "6.10 Power Considerations",
    "6.11 Design Application: Audio Amplifier",
    "6.12 Summary",
    "Chapter 7: Frequency Response",
    "7.1 Amplifier Frequency Response",
    "7.2 System Transfer Functions",
    "7.3 Frequency Response: Transistor Amplifiers with Circuit Capacitors",
    "7.4 Frequency Response: Bipolar Transistor",
    "7.5 Frequency Response: The FET",
    "7.6 High-Frequency Response of Transistor Circuits",
    "7.7 Design Application: A Two-Stage Amplifier with Coupling Capacitors",
    "7.8 Summary",
    "Chapter 8: Output Stages and Power Amplifiers",
    "8.1 Power Amplifiers",
    "8.2 Power Transistors",
    "8.3 Classes of Amplifiers",
    "8.4 Class-A Power Amplifiers",
    "8.5 Class-AB Push–Pull Complementary Output Stages",
    "8.6 Design Application: An Output Stage Using MOSFETs",
    "8.7 Summary",
    "PROLOGUE II: PROLOGUE TO ELECTRONIC DESIGN",
    "Design Approach",
    "System Design",
    "Electronic Design",
    "Conclusion",
    "PART 2: ANALOG ELECTRONICS",
    "Chapter 9: Ideal Operational Amplifiers and Op-Amp Circuits",
    "9.1 The Operational Amplifier",
    "9.2 Inverting Amplifier",
    "9.3 Summing Amplifier",
    "9.4 Noninverting Amplifier",
    "9.5 Op-Amp Applications",
    "9.6 Operational Transconductance Amplifiers",
    "9.7 Op-Amp Circuit Design",
    "9.8 Design Application: Electronic Thermometer with an Instrumentation Amplifier",
    "9.9 Summary",
    "Chapter 10: Integrated Circuit Biasing and Active Loads",
    "10.1 Bipolar Transistor Current Sources",
    "10.2 FET Current Sources",
    "10.3 Circuits with Active Loads",
    "10.4 Small-Signal Analysis: Active Load Circuits",
    "10.5 Design Application: An NMOS Current Source",
    "10.6 Summary",
    "Chapter 11: Differential and Multistage Amplifiers",
    "11.1 The Differential Amplifier",
    "11.2 Basic BJT Differential Pair",
    "11.3 Basic FET Differential Pair",
    "11.4 Differential Amplifier with Active Load",
    "11.5 BiCMOS Circuits",
    "11.6 Gain Stage and Simple Output Stage",
    "11.7 Simplified BJT Operational Amplifier Circuit",
    "11.8 Diff-Amp Frequency Response",
    "11.9 Design Application: A CMOS Diff-Amp",
    "11.10 Summary",
    "Chapter 12: Feedback and Stability",
    "12.1 Introduction to Feedback",
    "12.2 Basic Feedback Concepts",
    "12.3 Ideal Feedback Topologies",
    "12.4 Voltage (Series–Shunt) Amplifiers",
    "12.5 Current (Shunt–Series) Amplifiers",
    "12.6 Transconductance (Series–Series) Amplifiers",
    "12.7 Transresistance (Shunt–Shunt) Amplifiers",
    "12.8 Loop Gain",
    "12.9 Stability of the Feedback Circuit",
    "12.10 Frequency Compensation",
    "12.11 Design Application: A MOSFET Feedback Circuit",
    "12.12 Summary",
    "Chapter 13: Operational Amplifier Circuits",
    "13.1 General Op-Amp Circuit Design",
    "13.2 A Bipolar Operational Amplifier Circuit",
    "13.3 CMOS Operational Amplifier Circuits",
    "13.4 BiCMOS Operational Amplifier Circuits",
    "13.5 JFET Operational Amplifier Circuits",
    "13.6 Design Application: A Two-Stage CMOS Op-Amp to Match a Given Output Stage",
    "13.7 Summary",
    "Chapter 14: Nonideal Effects in Operational Amplifier Circuits",
    "14.1 Practical Op-Amp Parameters",
    "14.2 Finite Open-Loop Gain",
    "14.3 Frequency Response",
    "14.4 Offset Voltage",
    "14.5 Input Bias Current",
    "14.6 Additional Nonideal Effects",
    "14.7 Design Application: An Offset Voltage Compensation Network",
    "14.8 Summary",
    "Chapter 15: Applications and Design of Integrated Circuits",
    "15.1 Active Filters",
    "15.2 Oscillators",
    "15.3 Schmitt Trigger Circuits",
    "15.4 Nonsinusoidal Oscillators and Timing Circuits",
    "15.5 Integrated Circuit Power Amplifiers",
    "15.6 Voltage Regulators",
    "15.7 Design Application: An Active Bandpass Filter",
    "15.8 Summary",
    "PROLOGUE III: PROLOGUE TO DIGITAL ELECTRONICS",
    "Logic Functions and Logic Gates",
    "Logic Levels",
    "Noise Margin",
    "Propagation Delay Times and Switching Times",
    "PART 3: DIGITAL ELECTRONICS",
    "Chapter 16: MOSFET Digital Circuits",
    "16.1 NMOS Inverters",
    "16.2 NMOS Logic Circuits",
    "16.3 CMOS Inverter",
    "16.4 CMOS Logic Circuits",
    "16.5 Clocked CMOS Logic Circuits",
    "16.6 Transmission Gates",
    "16.7 Sequential Logic Circuits",
    "16.8 Memories: Classifications and Architectures",
    "16.9 RAM Memory Cells",
    "16.10 Read-Only Memory",
    "16.11 Data Converters",
    "16.12 Design Application: A Static CMOS Logic Gate",
    "16.13 Summary",
    "Chapter 17: Bipolar Digital Circuits",
    "17.1 Emitter-Coupled Logic (ECL)",
    "17.2 Modified ECL Circuit Configurations",
    "17.3 Transistor–Transistor Logic",
    "17.4 Schottky Transistor–Transistor Logic",
    "17.5 BiCMOS Digital Circuits",
    "17.6 Design Application: A Static ECL Gate",
    "17.7 Summary",
}
    
