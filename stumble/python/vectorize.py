import csv
from itertools import islice
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
from sklearn import metrics, preprocessing, cross_validation
import pandas as pd
import numpy as np

trainfnm = '../data/Train.tsv'
trainvecfnm = '../data/train_vec.npy'

stops = set(stopwords.words('english'))

def tokenize(text):
    # Basic tokenizing
    tokens = nltk.word_tokenize(nltk.clean_html(text))
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha() and t not in stops]

    # Stemming
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]

    # Add bigrams (as strings rather then tuples)
    # Not necessary because tdidf does it automatically
    #bigrams = nltk.bigrams(tokens)
    #tokens += [" ".join(bigram) for bigram in bigrams]

    # back to string so tfidf can parse it again
    return(" ".join(tokens))


def boiler_stream(filename, stop=None):
    with open(filename) as csvfile:
        next(csvfile, None)  # skip header        
        for line in islice(csv.reader(csvfile, delimiter='\t', quotechar='"'), stop):                
            tokens = tokenize(line[2])
            yield tokens

def vectorize(n):
    tfv = TfidfVectorizer(min_df=1, strip_accents='unicode', ngram_range=(1,2), stop_words='english',
        sublinear_tf=True, use_idf=True, smooth_idf=True)
    X = tfv.fit_transform(boiler_stream(trainfnm, n))
    np.save(trainvecfnm, X)

def train():
    
    logit = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
        intercept_scaling=1.0, class_weight=None, random_state=None)

    print "Load vectorised data"
    X = np.load(trainvecfnm)[None][0]
    N = X.shape[0]
    print "# docs: ", N
    
    print "Extracting labels"
    y = np.array(pd.read_table(trainfnm).iloc[:N,-1])
    
    print "Cross-validating..."
    mu = np.mean(cross_validation.cross_val_score(logit, X, y, cv=20, scoring='roc_auc'))
    print "20 Fold CV Score: ", mu

    logit.fit(X, y)
    ypred = logit.predict(X)

    print "Train accuracy: ", metrics.accuracy_score(y, ypred)
    print "Confusion: " 
    print metrics.confusion_matrix(y, ypred)    

    return {"X":X, "m":logit}

if __name__=="__main__":
    for t in boiler_stream(trainfnm, 10):
        print t
