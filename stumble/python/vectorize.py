import csv
from itertools import islice
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import metrics, preprocessing, cross_validation, linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np

datadir = '../data/'
trainfnm = datadir + 'Train.tsv'
testfnm = datadir + 'Test.tsv'
trainvecfnm = datadir + 'train_vec.npy'
testvecfnm = datadir + 'test_vec.npy'

stops = set(stopwords.words('english'))

def tokenize(text, addBigrams=False):
    # Basic tokenizing
    tokens = nltk.word_tokenize(nltk.clean_html(text))
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha() and t not in stops]

    # Stemming
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]

    # Add bigrams (as strings rather then tuples)
    # Not necessary because tdidf does it automatically
    if addBigrams:
        bigrams = nltk.bigrams(tokens)
        tokens += [" ".join(bigram) for bigram in bigrams]

    # back to string so tfidf can parse it again
    return(" ".join(tokens))


def boiler_stream(filename, stop=None):
    with open(filename) as csvfile:
        next(csvfile, None)  # skip header        
        for line in islice(csv.reader(csvfile, delimiter='\t', quotechar='"'), stop):                
            tokens = tokenize(line[2])
            yield tokens


def vectorize(n, comp=0):
    tfv = TfidfVectorizer(min_df=1, strip_accents='unicode', ngram_range=(1,2), stop_words='english',
        sublinear_tf=True, use_idf=True, smooth_idf=True)

    # Fit and transform
    X = tfv.fit_transform(boiler_stream(trainfnm, n))
    lsa = None
    scaler = None
    if comp > 0:
        lsa = TruncatedSVD(comp)
        scaler = Normalizer(copy=False)
        X = lsa.fit_transform(X)
        X = scaler.fit_transform(X)

    # Transform only
    Z = tfv.transform(boiler_stream(testfnm, n))
    if lsa:
        Z = lsa.transform(Z)
        Z = scaler.transform(Z)
    
    np.save(trainvecfnm, X)
    np.save(testvecfnm, Z)


def auroc(y, yp):
    fpr, tpr, thresholds = metrics.roc_curve(y, yp, pos_label=1)
    return metrics.auc(fpr,tpr)


def train():

    seed = 42
    
    logit = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
        intercept_scaling=1.0, class_weight=None, random_state=None)

    print "Load vectorised data"
    X = np.load(trainvecfnm)[None][0]
    Z  = np.load(testvecfnm)[None][0]
    N, M = X.shape
    Nt, Mt = Z.shape
    print "# train docs: ", N, " attrs: ", M
    print "# test docs: ", Nt, " attrs: ", Mt

    print "Extracting labels"    
    y = np.array(pd.read_table(trainfnm).iloc[:N,-1])

    # Create validation set
    Xtr, Xva, ytr, yva = train_test_split(X, y, train_size=0.8, random_state=seed)
    
    print "Cross-validating on whole set..."
    mu = np.mean(cross_validation.cross_val_score(logit, X, y, cv=10, scoring='roc_auc'))
    print "20 Fold CV Score: ", mu

    print "Validating..."
    logit.fit(Xtr, ytr)
    pred_yva = logit.predict(Xva)
    print "Validation auroc: ", auroc(yva, pred_yva)
    
    print "training on full data"
    logit.fit(X, y)

    # Train accuracy
    ypred = logit.predict(X)
    print "Train accuracy: ", metrics.accuracy_score(y, ypred)
    print "Train auroc: ", auroc(y, ypred)
    print "Confusion: " 
    print metrics.confusion_matrix(y, ypred)    

    # Predict test set
    yp = logit.predict(Z)
    urlids = pd.read_table(testfnm)['urlid']
    df = pd.DataFrame(data=urlids)
    df['label'] = yp
    df.to_csv(datadir + 'submission.csv', index=False)
    print "submission file created.."

    return {"X":X, "m":logit}

if __name__=="__main__":
    for t in boiler_stream(trainfnm, 10):
        print t
