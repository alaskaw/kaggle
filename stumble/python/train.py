# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing, cross_validation
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_selection import chi2, f_classif, SelectPercentile, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.externals import joblib # For caching of intermediate pipeline results

from tokenize import tokenize_and_stem
from transform import *
from timer import Timer

datadir = '../data/'
trainfnm = datadir + 'Train.tsv'
testfnm = datadir + 'Test.tsv'

VEC_ONLY = True

# Tf-idf with optional dimensionality reduction
def get_vec_pipe(num_comp=0, reducer='svd', stem=1):
    ''' Create text vectorization pipeline with optional dimensionality reduction. '''

    if stem:
        tfv = TfidfVectorizer(
            min_df=6, max_features=None, strip_accents='unicode',
            analyzer="word", tokenizer=tokenize_and_stem, ngram_range=(1, 2),
            use_idf=1, smooth_idf=1, sublinear_tf=1)
    else:
        tfv = TfidfVectorizer(
            min_df=6, max_features=None, strip_accents='unicode',
            analyzer="word", token_pattern=r'\w{1,}', ngram_range=(1, 2),
            use_idf=1, smooth_idf=1, sublinear_tf=1)

    # Vectorizer
    vec_pipe = [
        ('json_extr', JsonFields(0, ['body', 'title', 'url'])),
        ('text_vec', Pipeline([
            ('squash', Squash()),
            ('vec', tfv)
            ])
        )
    ]

    # Reduce dimensions of tfidf
    if num_comp > 0:
        if reducer == 'svd':
            red_pipe = Pipeline([
                ('svd', TruncatedSVD(num_comp)),
                ('svd_norm', Normalizer(copy=True))
            ])
        elif reducer == 'kbest':
            red_pipe = Pipeline([
                ('kbest', SelectKBest(chi2, k=num_comp)),
                ('norm', Normalizer(copy=True))
            ])
        elif reducer == 'percentile':
            red_pipe = Pipeline([
                ('kbest', SelectPercentile(f_classif, percentile=10)),
                ('norm', Normalizer(copy=True))
            ])
        vec_pipe.append(('reduce', red_pipe))

    vec_pipe = Pipeline(vec_pipe)
    return vec_pipe


# Complete transform chain
def get_trf_chain(vec_only, num_alch_cat, num_comp=0, reducer='svd', stem=0):
    ''' Return a complete feature extraction/creation pipeline '''

    vec_pipe = get_vec_pipe(num_comp, reducer, stem)
    if vec_only:
        return [('union', FeatureUnion([('boil_vec', vec_pipe)]))]

    # Article length
    length_pipe = Pipeline([
        ('length', Length(0)),
        ('norm', StandardScaler(with_mean=False)),
    ])

    # Alchemy category
    alc_pipe = Pipeline([
        ('sel_alc', Select(1)),
        ('one_hot', OneHotEncoder(n_values=num_alch_cat))
    ])

    # Is news?
    news_pipe = Pipeline([
        ('news_bin', Select(2)),
        ('news_float', Float())
    ])

    chain = [
        ('union', FeatureUnion([
            ('boil_vec', vec_pipe),
            ('boil_length', length_pipe),
            ('alc_cat', alc_pipe),
            ('news_bin', news_pipe)
            ])
        )
    ]

    #chain.append(('rep', DatasetReporter()))
    return chain


def preprocess(X):
    ''' Preprocess a data frame (upfront). '''

    # Alchemy category
    X['alchemy_category'] = X['alchemy_category'].fillna('unknown') # get rid of Nan
    le_alc = LabelEncoder()
    cat = X['alchemy_category']
    le_alc.fit(cat)
    X['alchemy_factor'] = le_alc.transform(cat)

    # is news binary category
    X['is_news'].fillna(0, inplace=True)
    X['is_news'] = X['is_news'].astype(int)

    # Filter columns
    cols = ['boilerplate', 'alchemy_factor', 'is_news']
    return X[cols]


def get_estimator_pipe(name, model, vec_only, num_alch_cat, lsa_comp, reducer, stem):
    ''' Concatenate a transform chain and a classifier. '''
    chain = get_trf_chain(vec_only, num_alch_cat, lsa_comp, reducer, stem)
    chain.append((name, model))
    return Pipeline(chain)


def build_ensemble(model_list, estimator):
    ''' Build an ensemble as a FeatureUnion of ModelTransformers and a final estimator using their
        predictions as input. '''

    models = []
    for i, model in enumerate(model_list):
        models.append(('model_transform'+str(i), ModelTransformer(model)))

    features = FeatureUnion(models)
    ensemble = Pipeline([
        ('features', features),
        ('estimator', estimator)
    ])
    return ensemble



def load():
    ''' Load and return train data, labels and test data. '''
    X = pd.read_table(trainfnm, na_values=["?"])        # Full training set
    y = X['label']
    X = preprocess(X)
    X_test = pd.read_table(testfnm, na_values=["?"])    # Test set
    X_test = preprocess(X_test)
    return X, y, X_test


def describe(X, y):
    ''' Return statistical overview of a pandas data frame. '''
    X['label'] = y
    X.groupby('label').describe()
    X.hist(column='is_news', by='label', bins=50)


def build_all_classifiers():
    ''' Return a dictionary with clf name as key, and model plus grids as values. '''

    logit = LogisticRegression(
        penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
        intercept_scaling=1.0, class_weight=None, random_state=None) # random_state = random_seed
    logit_grid_c = {'logit__C' : (0.1, 1, 10), 'union__boil_vec__text_vec__vec__ngram_range': [(1, 2), (1, 3)]}
    logit_grid_s = {'logit__C' : (0.1, 1, 10)}
    logit_grid_b = {'logit__C' : [(1)]}
    logit_clf = {'model':logit, 'grid_c':logit_grid_c, 'grid_s':logit_grid_s, 'grid_b':logit_grid_b}

    sgd = SGDClassifier(loss='hinge', penalty='elasticnet', l1_ratio=0.5)
    sgd_grid_c = {'sgd__l1_ratio': (0, 0.25, 0.5,1), 'union__boil_vec__text_vec__vec__ngram_range': [(1, 2), (1, 3)]}
    sgd_grid_s = {'sgd__l1_ratio': (0, 0.25, 0.5, 1)}
    sgd_grid_b = {'sgd__l1_ratio': [(0.0)]}
    sgd_clf = {'name':'sgd', 'model':sgd, 'grid_c':sgd_grid_c, 'grid_s':sgd_grid_s, 'grid_b':sgd_grid_b}

    svc = SVC(kernel='linear', probability=True)
    svc_grid_c = {'svc__C': [0.1, 1, 10, 100], 'union__boil_vec__text_vec__vec__ngram_range': [(1, 2), (1, 3)]}
    svc_grid_s = {'svc__C': [1, 10]}
    svc_grid_b = {'svc__C': [1]}
    svc_clf = {'name':'svc', 'model':svc, 'grid_c':svc_grid_c, 'grid_s':svc_grid_s, 'grid_b':svc_grid_b}

    knn = KNeighborsClassifier(n_neighbors=5)
    knn_grid_c = {'knn__n_neighbors': [5, 10, 25], 'union__boil_vec__text_vec__vec__ngram_range': [(1, 2), (1, 3)]}
    knn_grid_s = {'knn__n_neighbors': [10, 25]}
    knn_grid_b = {'knn__n_neighbors': [25]}
    knn_clf = {'name':'knn', 'model':knn, 'grid_c':knn_grid_c, 'grid_s':knn_grid_s, 'grid_b':knn_grid_b}

    classifiers = {'logit':logit_clf, 'sgd':sgd_clf, 'svc':svc_clf}#, 'knn':knn_clf}
    return classifiers


def build_all_pipes(grid, vec_only, num_alch_cat, red_comp, reducer, stem):
    ''' Combine classifiers with transform pipes. '''
    clfs = build_all_classifiers()
    pipes = []
    for name, clf in clfs.iteritems():
        pipe = get_estimator_pipe(name, clf['model'], vec_only, num_alch_cat, red_comp, reducer, stem)
        pipe.grid = clf[grid]
        pipe.name = name
        pipes.append(pipe)
    return pipes


def build_simple_pipes(grid):
    ''' Create classifier-only pipes (without prior transforms, if this is done upfront e.g.). '''
    clfs = build_all_classifiers()
    pipes = []
    for name, clf in clfs.iteritems():
        pipe = Pipeline([(name, clf['model'])])
        pipe.grid = clf[grid]
        pipe.name = name
        pipes.append(pipe)
    return pipes


def grid_search(trf_in_fold=False, grid='grid_s', num_folds=5, reducer='svd', red_comp=0):
    """ Return best estimator from parameter grid. """

    random_seed = 42
    num_jobs = 5
    scoreFn = 'roc_auc'

    X, y, X_test = load()
    num_alch_cat = len(np.unique(X['alchemy_factor']))

    if not trf_in_fold:
        print "Transforming whole data set prior to training:"
        print "Fit and transform train set..."
        trf_pipe = Pipeline(get_trf_chain(VEC_ONLY, num_alch_cat, red_comp, reducer))
        X = trf_pipe.fit_transform(X)
        print "Transform only test set..."
        X_test = trf_pipe.transform(X_test)
        print "Building non-transforming pipes..."
        pipes = build_simple_pipes(grid)
    else:
        print "Building in-fold transforming pipes..."
        pipes = build_all_pipes(grid, VEC_ONLY, num_alch_cat, red_comp, reducer)

    # Split off a validation set for comparison of different models
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Now fit the pipelines
    best = []
    best_score = []
    print "\n\n"
    for i in range(len(pipes)):
        pipe = pipes[i]

        print "Hypertuning model ", i+1 , " out of ", len(pipes), ": ", pipe.name
        print "================================================================================"

        # Uses stratified k-fold by default supposedly
        gs = GridSearchCV(pipe, pipe.grid, scoring=scoreFn, cv=num_folds, n_jobs=num_jobs, verbose=2)

        with Timer(pipe.name + " fitting"):
            model = gs.fit(X_train, y_train)

        print "Best score on training set (CV): %0.3f" % gs.best_score_
        print "Best parameters set:"
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(pipe.grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        for params, mean_score, scores in gs.grid_scores_:
            print "%0.4f (+/-%f) for %r: %r" % (mean_score, scores.std() / 2, params, scores)

        # Assess and predict (validation error etc.)
        print "\nPredict: "
        yp_val = model.predict(X_val)
        diff = y_val - yp_val
        print "Auroc val:", vectorize.auroc(y_val, yp_val)
        print classification_report(y_val, yp_val)
        metrics.confusion_matrix

        best.append(gs.best_estimator_)
        print "10-fold cross-validation of best instance on whole set..."
        tr_roc = np.mean(cross_validation.cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='roc_auc', verbose=2))
        best_score.append(tr_roc)
        print "Mean score: ",  tr_roc
        print "================================================================================"

    return best, best_score


def train(trf_in_fold=False, single_clf=None):
    ''' Train best or ensemble of best from grid search. '''

    random_seed = 42
    num_jobs = 5
    reducer = 'svd'
    red_comp = 0
    stem = 0
    grid = 'grid_b'
    scorer = 'roc_auc'

    X, y, X_test = load()
    test_urlids = pd.read_table(testfnm, na_values=["?"])['urlid'] # Store for later creating df of predictions
    num_alch_cat = len(np.unique(X['alchemy_factor']))

    if not trf_in_fold:
        print "Transforming whole data set prior to training:"
        print "Fit and transform train set..."
        trf_pipe = Pipeline(get_trf_chain(VEC_ONLY, num_alch_cat, red_comp, reducer, stem))
        X = trf_pipe.fit_transform(X)
        print "Transform only test set..."
        X_test = trf_pipe.transform(X_test)
        print "Building non-transforming pipes..."
        pipes = build_simple_pipes(grid)
    else:
        print "Building in-fold transforming pipes..."
        pipes = build_all_pipes(grid, VEC_ONLY, num_alch_cat, red_comp, reducer, stem)

    if single_clf:
        print "Using selected estimator only: "
        for pipe in pipes:
            if pipe.name == single_clf:
                print pipe.name
                model = pipe
    else:
        print "Building ensemble"
        model = build_ensemble(pipes, LogisticRegression())

    print "Running cross-validation of final model on whole set: "
    final_roc = np.mean(cross_validation.cross_val_score(model, X, y, cv=10, scoring=scorer, verbose=2))
    print "Ensemble mean auroc: ", final_roc

    # Train final model on whole dataset
    print "Fitting final model to whole training set for prediction:"
    model.fit(X, y)
    yp = model.predict_proba(X_test)[:,1]
    df = pd.DataFrame(data=test_urlids, columns=["urlid"])
    df['label'] = yp
    df.to_csv(datadir + 'submission.csv', index=False)
    print "Done. Submission file written to data dir."


if __name__ == "__main__":
    train()
