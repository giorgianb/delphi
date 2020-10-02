from typing import Dict, Tuple, Set, Iterable
import numpy as np
import nltk.corpus
from collections import defaultdict
import gensim.downloader
import gensim.models.word2vec
import abc
import sklearn.decomposition
import hashlib

Word = str
Sentence = Tuple[Word]

class Document:
    def __init__(self, sentences: Tuple[Sentence]):
        self._sentences = sentences

    @property
    def sentences(self) -> Tuple[Sentence]:
        return self._sentences

    @property
    def words(self) -> Tuple[Word]:
        return tuple(word for sentence in self._sentences for word in sentences)


class TFIDF:
    def __self__(self, documents: Set[Document]):
        self._documents = documents
        # compute the idfs
        # we need to compute the number of documents that contain the word w
        dcount = collections.defaultdict(int)
        for document in documents:
            for word in set(document.words):
                dcount[word] += 1

        self._idf_dict = {}
        for word, count in dcount.items():
            self._idf_dict[word] = np.log(N/count)

    def __call__(self, word: str, document: Document) -> float:
        count = document.words.count(word) / len(document)

        return count * self._idf_dict[word]

    @property
    def known_words(self) -> Set[Word]:
        return self._idf_dict.keys()

class Indicator:
    def __init__(self, vocab: Set[Word]):
        self._vocab = dict(zip(vocab, range(len(vocab))))

    def words(self) -> Iterable[str]:
        return self._vocab.keys()

    def size(self) -> int:
        return len(self._vocab) + 1

    def __call__(self, word: str) -> np.ndarray:
        v = np.zeros(self.size() + 1)
        i = self._vocab[word] if word in self._vocab else len(self._vocab)
        v[i] = 1

        return v

class SentenceEmbedding(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sentence: Sentence) -> np.ndarray:
        return np.zeros(1)

class BagOfWords(SentenceEmbedding):
    def __init__(self, documents: Set[Document]):
        self._tfidf = TFIDF(documents)
        self._indicator = Indicator(self._tfidf.known_words)

    def __call__(self, sentence: Sentence) -> np.ndarray:
        return sum(map(lambda word: self._ind(word) * self._tfidf(word), sentence))

class BloomFilter(SentenceEmbedding):
    def __init__(self, n_hash_functions: int, bit_array_size: int):
        self._k = n_hash_functions
        self._m = bit_array_size

    def __call__(self, sentence: Sentence) -> np.ndarray:
        v = np.zeros(self._m)

        for word in sentence:
            for i in range(self._k):
                word = word + str(i)
                h = hashlib.sha256(word.encode("UTF-8"))
                p = int(h.hexdigest(), 16) % self._m
                assert int(h.hexdigest(), 16) >= self.m
                v[p] += 1

        return v

class Sentence2Vec(SentenceEmbedding):
    corpus = None
    word2vec = None

    def __init__(self, alpha: float, documents: Set[Document]):
        if Sentence2Vec.corpus is None:
            Sentence2Vec.corpus = gensim.downloader.load('text8')

        if Sentence2Vec.word2vec is None:
            Sentence2Vec.word2vec = gensim.models.word2vec.Word2Vec(corpus)


        self._alpha = alpha
        self._word_counts = collections.defaultdict(int)
        self._total_words = 0

        for document in documents:
            for word in document.words:
                self._word_counts[word] += 1
                self._total_words += 1

        self._rank = sum(1 for document in documents for sentence in document)

        train_matrix = []
        weight_fn = lambda word: self._alpha/(self._alpha + self.word_probability(word))
        vector_fn = lambda word: Sentence2Vec.wv[word]
        for sentence in document.sentences:
            rank = len(document.sentences)
            words = document.words
            weights = map(weight_fn, words)
            vectors = map(vector_fn, words)
            v = sum(map(lambda wv: wv[0] * wv[1], zip(weights, vectors)))/self._rank
            train_matrix.append(v)

        pca = sklearn.decomposition.PCA(n_components=1) 
        pca.fit(train_matrix)
        fpc = pca.components[0]
        
        self._fpc_weight = fpc.dot(fpc)

    def word_probability(self, word: Word) -> float:
        return self._word_counts[word]/self._total_words

    def __call__(self, sentence: Sentence) -> np.ndarray:
        weight_fn = lambda word: self._alpha/(self._alpha + self.word_probability(word))
        vector_fn = lambda word: Sentence2Vec.wv[word]

        weights = map(weight_fn, words)
        vectors = map(vector_fn, words)
        v = sum(map(lambda wv: wv[0] * wv[1], zip(weights, vectors)))/self._rank
