import requests
import json
import csv
import stanza
import nltk.corpus
import sys
import string
import stanza
import string

nlp = stanza.Pipeline('en', processors='tokenize,lemma,pos')
def tokenize_and_lemmatize_sentences(data):
    global nlp
    stop_words = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation)
    rows = []
    texts = []

    data = tuple(data)
    for i, text in enumerate(data):
        doc = nlp(text)
        doc_sentences = doc.sentences
        sentences = [[word.lemma for word in sent.words if word.lemma not in stop_words] 
                for sent in doc_sentences]
        sentences_text = [[word.text for word in sent.words] for sent in doc_sentences]
        rows.append(sentences)
        texts.append(sentences_text)

    return rows, texts

def tokenize_and_lemmatize_data(data):
    stop_words = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation)
    rows = []
    texts = []

    data = tuple(data)
    for i, (topic, text) in enumerate(data):
        print(f"[{i + 1}/{len(data)}] Parsing {topic}")
        doc = nlp(text)
        doc_sentences = tuple(
                filter(
                    lambda sent: any(word.pos == 'VERB' for word in sent.words), 
                    doc.sentences
                    )
                )
        sentences = [[word.lemma for word in sent.words if word.lemma not in stop_words] 
                for sent in doc_sentences]
        sentences_text = [[word.text for word in sent.words] for sent in doc_sentences]
        rows.append(json.dumps(sentences))
        texts.append(json.dumps(sentences_text))

    return rows, texts


if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]
    with open(input_file, newline='') as f:
        fin = csv.reader(f)
        titles_texts = list(fin)

    titles = tuple(item[0] for item in titles_texts)
    texts = tuple(item[1] for item in titles_texts)
    parsed, texts = tokenize_and_lemmatize_data(zip(titles, texts))
    print("Total number of topics:", len(titles))
    with open('data.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(zip(titles, parsed, texts))
