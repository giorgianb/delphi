import os
os.chdir('../')

import requests
import json
import csv
import stanza
import nltk.corpus
url_template = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles={}&prop=extracts&explaintext"


#titles = ("Semiconductor", "Biology", "Operational_amplifier", "The_Doors", 0)
titles = (
        "The_Doors", 
        "The_Beatles", 
        "The_Who", 
        "Talking_Heads", 
        "Jimi_Hendrix", 
        "Bob_Marley",
        "Semiconductor",
        "Biology",
        "Operational_amplifier",
        "Real_number",
        "Integer",
        "Stony_Brook_University",
        "Jean-Paul_Sartre",
        "Georg_Wilhelm_Friedrich_Hegel",
        "Plato",
        "Sigma_Phi_Delta",
        "Socrates",
        "Karl_Marx",
        "Romania",
        "French_Revolution",
        "Greece",
        "Mihai_Eminescu",
        "Indo-European_languages",
        # from the ten longest articles
        "1996_California_Proposition_218",
        "South_African_labour_law",
        # from wikipedia daily articles
        "Episode_14_(Twin_Peaks)",
        "Typhoon_Gay_(1989)",
        "Dali_(goddess)",
        "William_Henry_Harrison_1840_presidential_campaign",
        "Loveless_(album)",
        "Henry_Clifford,_10th_Baron_Clifford",
        "1981_UEFA_Cup_Final",
        "Hellraiser:_Judgment",
        "King_brown_snake",
        "Japanese_battleship_Yashima",
        "Fabian_Ware",
        "St._Croix_macaw",
        "Edward_Thomas_Daniell",
        "Project_Excalibur",
        "Rwandan_Civil_War",
        "Vermilion_flycatcher",
        "Antiochus_XI_Epiphanes",
        "Joseph_A._Lopez",
        "SMS_Dresden_(1907)",
        "Banksia_petiolaris",
        "Cape_Feare",
        "Florin_(British_coin)",
        "Hassium",
        "Battle_of_Cape_Ecnomus",
        "Jerome,_Arizona",
        "Hurricane_Gonzalo",
        "Mike_Capel",
        "Eastern_green_mamba",
        "Master_Chief_(Halo)"
)

texts = []
for i, title in enumerate(titles):
    print("[{}/{}] Fetching {}".format(i, len(titles), title))
    url = url_template.format(title)
    r = requests.get(url)
    data = r.json()
    page_texts = []
    for page in data['query']['pages']:
        page_texts.append(data['query']['pages'][page]['extract'])

    text = ''.join(page_texts)
    texts.append(''.join(page_texts))

def tokenize_and_lemmatize_data(data):
    nlp = stanza.Pipeline('en', processors='tokenize,lemma')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    rows = []
    texts = []

    data = tuple(data)
    for i, (topic, text) in enumerate(data):
        print("[{}/{}] Parsing {}".format(i, len(data), topic))
        doc = nlp(text)
        sentences = [[word.lemma for word in sent.words if word.lemma not in stop_words] 
                for sent in doc.sentences]
        sentence_text = [[word.text for word in sent.words] for sent in doc.sentences]
        rows.append(json.dumps(sentences))
        texts.append(json.dumps(sentence_text))

    return rows, texts

parsed, texts = tokenize_and_lemmatize_data(zip(titles, texts))

print("Total number of topics:", len(titles))
with open('data.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(zip(titles, parsed, texts))
