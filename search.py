import random
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import wordnet, pos_tag

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import WordNetLemmatizer
import pickle
import io
from tqdm import tqdm
from itertools import islice
import numpy as np


def load_vectors(fname, limit):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(islice(fin, limit), total=limit):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data


class Document:
    def __init__(self, title, text, answer):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.answer = answer

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' - ' + self.answer]


def build_index():
    df = pd.read_csv('dataset_no_links.csv')
    for i, row in df.iterrows():
        index_search.append(
            Document(row[' Category'][0] + row[' Category'][1:].lower(), row[' Question'], row[' Answer']))


def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN


def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                  for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                     for word, tag in pos_tagged])


def normalize_text(text):
    text = re.sub(html_pattern, '', text)
    text = re.sub(r'[^a-z ]+', r'', text.lower())
    text = my_lemmatizer(text)
    text = ' '.join([word for word in text.split() if not word in sw_eng])
    return text


def get_1k_most_relevant(query):
    normalized_query = normalize_text(query)
    splitted = normalized_query.split()
    score = {}
    for word in splitted:
        if word not in inverted_index.keys():
            continue
        for item in inverted_index[word].items():
            if item[0] not in score.keys():
                score[item[0]] = len(item[1])
            else:
                score[item[0]] += len(item[1])
    score = list(sorted(score.items(), key=lambda item: item[1]))
    score.reverse()
    return score[:1000]


def vectorize_text(text):
    splitted = text.split()
    return sum(list(map(lambda w: np.array(list(vecs.get(w, zero))), splitted))) / (
        len(splitted) if len(splitted) != 0 else 1)


def sort_1k_top(query):
    score = get_1k_most_relevant(query)
    query_vec = vectorize_text(normalize_text(query))
    rating = []
    for i, freq in score:
        rating.append(((np.average(vectorize_text(normalize_text(df[i])) - query_vec)) ** 2, i))
    rating.sort(key=lambda sci: sci[0])
    return rating


index_search = []
sw_eng = set(stopwords.words('english'))
a_file = open("invindex.pkl", "rb")
df = pd.read_csv('normalisedtext.csv').iloc[:, 1]
inverted_index = pickle.load(a_file)
html_pattern = re.compile(
    '<script[\s\S]*?/script>[\s\S]*?|href=[\s\S]*?>[\s\S]*?|<br />|<ul[\s\S]*?/ul>[\s\S]*?|<li[\s\S]*?/li>[\s\S]*?|<style type[\s\S]*?/style>[\s\S]*?|<object[\s\S]*?/object>[\s\S]*?|(<a href[\s\S]*?>[\s\S]*?)|(\b(http|https):\/\/.*[^ alt]\b)|</ul>|</li>|<br/>|<!--[\s\S]*?-->[\s\S]*?|<div style[\s\S]*?>[\s\S]*?|<img[\s\S]*?>[\s\S]*?|<div id[\s\S]*?>[\s\S]*?|<div class[\s\S]*?>[\s\S]*?|</object>|<embed[\s\S]*?/>[\s\S]*?|<param[\s\S]*?/>[\s\S]*?|<noscript>[\s\S]*?</noscript>[\s\S]*?|<link rel[\s\S]*?>[\s\S]*?|<p style="text-align: center;">|<iframe[\s\S]*?</iframe>[\s\S]*?')
vecs = load_vectors('crawl-300d-2M.vec', 300000)
zero = sum(vecs.values()) / len(vecs)
