from __future__ import print_function
import nltk, sklearn, string, os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from configparser import ConfigParser
import os
from string import punctuation
import pandas as pd

# read config file
config = ConfigParser()
config.read(r'datadir.ini')
direct = config.get('Path', 'Corpusdirectoy')


# Preprocessing text with NLTK package
# import nltk
token_dict = {}
stemmer = PorterStemmer()
# stemmer = ntlk.stem.SnowballStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
    # return tokens

def strip_punctuation(s):
    return''.join(c for c in s if c not in punctuation)

titles = []
titlespdf = []
for (subdir, dirs, files) in os.walk(direct):
    for i, f in enumerate(files):
        if f.endswith('.txt'):
            file_path = subdir + os.path.sep + f
            shakes = open(file_path, 'r', encoding='utf-8')
            text = shakes.read()
            no_punctuation = strip_punctuation(text)
            token_dict[f] = no_punctuation
            titles.append(f)

true_k =10

vectorizer = TfidfVectorizer(tokenizer=tokenize, encoding='utf-8', stop_words='english')
X = vectorizer.fit_transform(token_dict.values())
tfidf_matrix = vectorizer.fit_transform(token_dict.values())

###############################################################################
# Do the actual clustering
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1,verbose=0, random_state=3000, copy_x=True, n_jobs=1)# random_state=5000)
y = km.fit(X)
print(km)
print(y)

from sklearn.metrics import silhouette_samples, silhouette_score
cluster_labels = km.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", true_k,
          "The average silhouette_score is :", silhouette_avg)


# print("Top terms per cluster:")
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i, end='')
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind], end='')
#     print()
#


clusterlist = km.labels_.tolist()
# print(clusterlist)
documents = {'title': titles, 'cluster': clusterlist}
frame = pd.DataFrame(documents, index = [clusterlist] , columns = ['title', 'cluster'])

frame.to_csv('summary of clusters.csv')
