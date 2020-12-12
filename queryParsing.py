#Query_Parsing
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import json


# query = input("Enter Search Query: ")


def parsed_result(query):
    words = query.split()

    stop_words = set(stopwords.words("english"))
    # print(sorted(stop_words))
    corpus = []
    # Remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', str(query))

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
    corpus.append("")  # log 1=0 df cannot b 0
    # print(corpus)

    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1, 3))
    X = cv.fit_transform(corpus)
    # print(X)
    # print(cv)

    listOfURLs = []
    x = len(list(cv.vocabulary_.keys()))  # list(cv.vocabulary_.keys()) is a list of the words in query
    # print(list(cv.vocabulary_.keys())[:x])

    with open(r"static/database_Aparajita_Inverse_Indexed.json") as idx:
        dicts = json.load(idx)

    def get_top_n3_words(corpus):
        vec1 = CountVectorizer(ngram_range=(3, 3), max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec1.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq

    def get_top_n2_words(corpus):
        vec1 = CountVectorizer(ngram_range=(2, 2),
                               max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec1.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq

    def get_top_n1_words(corpus):
        vec1 = CountVectorizer(ngram_range=(1, 1),
                               max_features=2000).fit(corpus)
        bag_of_words = vec1.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vec1.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq

    # creating list of tuples for each word
    grams = []
    if len(words) >= 3:
        grams.append(get_top_n3_words(corpus))  # storing trigrams as 0 index first priority
    if len(words) >= 2:
        grams.append(get_top_n2_words(corpus))  # storing bigrams
    if len(words) > 0:
        grams.append(get_top_n1_words(corpus))  # storing 1 word

    k = -1
    for j in grams:
        # print(j)
        for i in j:
            # print(i)
            listOfURLs.append([])
            # print(i[0])
            if i[0] in dicts:
                k = k + 1
                listOfURLs[k].append(dicts[i[0]])
                # print(listOfURLs[k])
            listOfURLs[k].sort(key=lambda tup: (tup[0], tup[1]), reverse=True)
            # print(listOfURLs[k])
            if len(listOfURLs[
                       k]) != 0:  # if trigram exist in dicts do not use bigram or 1 words anymore to avoid wrong
                # results
                break
            # print(listOfURLs[k])
        if len(listOfURLs[k]) != 0:
            break

    finalList = []
    for i in listOfURLs:
        for j in i:
            for l in j:
                # finalList.append(l[2])
                dictionary = dict([("url", l[2]), ("title", l[3]), ("abstract", l[4])])
                finalList.append(dictionary)

    return finalList
