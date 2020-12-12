import os
# usr = os.getlogin()
# os.chdir('/Users/'+usr+'/Desktop')
# cwd = os.getcwd()
# print('Working in ', cwd, '\n')

import pandas

dataset = pandas.read_csv('output.csv', delimiter=',')


import re
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words("english"))

corpus = []
dataset['word_count'] = dataset["Title"].apply(lambda x: len(str(x).split(" ")))
ds_count = len(dataset.word_count)
for i in range(0, ds_count):
    # Remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', str(dataset["Title"][i])+' '+str(dataset["Abstract"][i]))

    # Convert to lowercase
    text = text.lower()

    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    # Convert to list from string
    text = text.split()

    # Stemming
    ps = PorterStemmer()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
                                                        stop_words]
    text = " ".join(text)
    corpus.append(text)

# Tokenize the text and build a vocabulary of known words
from sklearn.feature_extraction.text import CountVectorizer
import re

cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))
X = cv.fit_transform(corpus)

# Get TF-IDF (term frequency/inverse document frequency) --
# TF-IDF lists word frequency scores that highlight words that 
# are more important to the context rather than those that 
# appear frequently across documents

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)

# Get feature names
feature_names = cv.get_feature_names()

# Fetch document for which keywords needs to be extracted


from scipy.sparse import coo_matrix


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items):
    # Use only topn items from vector
    # sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # Word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # Keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # Create tuples of feature,score
    # Results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results


results = []
for doc in corpus:
    # Generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    # Sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    
    keywords = extract_topn_from_vector(feature_names, sorted_items)

    results.append(keywords)

import json

with open('keywords.json', 'w') as fout:
    json.dump(results, fout)
