#compiled different resources to run a classification problem on
#words in 'why do you want to go to the dinner?' question

#60% accuracy

import nltk
import string
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

from dukeC3 import init, dict_sorter
from collections import Counter
import numpy as np

data_spring2018 = init()['spring2018'][0]
data_fall2018 = init()['fall2018'][0]
data_fall2017 = init()['fall2017'][0]

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
s = re.compile(r' | '.join(STOPWORDS))



#basic preparation of text
def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = re.sub (' +', ' ', text)
    text = ' '.join([n for n in text.split(' ') if n not in STOPWORDS])
    return text
    
#attempt at lemmatize and tokenizing with nltk
wnl = WordNetLemmatizer()
english_stopwords = nltk.corpus.stopwords.words('english')
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(token):
    token,tb_pos = nltk.pos_tag([token])[0]
    pos = get_wordnet_pos(tb_pos)
    lemma = wnl.lemmatize(token,pos)
    return lemma

all_stopwords = english_stopwords + list(string.punctuation) + ['\"']

def text_preprocessor(text):
    text = BAD_SYMBOLS_RE.sub('', text)
    tokens = nltk.wordpunct_tokenize(text)
    clean_tokens = []
    for t in tokens:
        if t.lower() not in all_stopwords and len(t) > 2:
            clean_tokens.append(lemmatizer(t.lower()))
    return ' '.join(clean_tokens)

#try to alternate between text_prepare - raw words - and text_preprocessor - lemmatized words

X_val = [text_prepare(n) for n in data_spring2018['why']]
y_val = np.array(data_spring2018['accepted'].fillna(0))
X_train = [text_prepare(n) for n in data_fall2018['why']]
y_train = np.array(data_fall2018['accepted'].fillna(0))
    
words_counts = Counter([n for b in X_train for n in b.split(' ')])
tags_counts = Counter([b for b in y_train])
wc = dict_sorter(words_counts)

DICT_SIZE = 2000
WORDS_TO_INDEX = {wc[n][0]: n for n in range(DICT_SIZE)}
ALL_WORDS = WORDS_TO_INDEX.keys()


def my_bag_of_words(text, words_to_index, dict_size):
    result_vector = np.zeros(dict_size)
    for n in text.split(' '):
        try:
            result_vector[words_to_index[n]] += 1
        except:
            continue
    return result_vector
    
X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)


def tfidf_features(X_train, X_val):
    tfidf_vectorizer = TfidfVectorizer(max_df = .9, min_df = 5, token_pattern = '(\S+)', ngram_range = (1, 2))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    return X_train, X_val, tfidf_vectorizer.vocabulary_
    
X_train_tfidf, X_val_tfidf, tfidf_vocab = tfidf_features(X_train, X_val)


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.fit_transform(y_val)


def train_classifier(X_train, y_train):

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    ovs = OneVsRestClassifier(LogisticRegression())
    ovs.fit(X_train, y_train)

    return ovs

classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print(y_val_predicted_labels_tfidf)
y_val_pred_inversed = lb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = lb.inverse_transform(y_val)

def print_evaluation_scores(y_val, predicted):
	print accuracy_score(y_val, predicted)
	print f1_score(y_val, predicted, average='macro')
	print f1_score(y_val, predicted, average='micro')
	print f1_score(y_val, predicted, average='weighted')  

print 'Bag-of-words'
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print 'Tfidf'
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
