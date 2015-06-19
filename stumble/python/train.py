# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing, cross_validation
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import chi2, f_classif, SelectPercentile, SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib # For caching of intermediate pipeline results

from tokenizer import tokenize_and_stem
from transform import *
from timer import Timer

from sets import Set
from pprint import pprint

datadir = '../data/'
trainfnm = datadir + 'Train.tsv'
testfnm = datadir + 'Test.tsv'

VEC_ONLY = True


def auroc(y, yp):
    fpr, tpr, thresholds = metrics.roc_curve(y, yp, pos_label=1)
    return metrics.auc(fpr, tpr)


def get_custom_pipe(num_comp=0, stem=1, clf=None):
    ''' Create text vectorization pipeline with optional dimensionality reduction. '''

    # Get non-dim-reduced vectorizer
    pipe = get_vec_pipe(num_comp=0, stem=stem)

    # Add a logit on non-reduced tfidf, and ensemble on reduced tfidf
    clfs = ['rf', 'sgd', 'gbc']
    pipe.steps.append(
        ('union', FeatureUnion([
            ('logit', ModelTransformer(build_classifier('logit'))),
            ('featpipe', Pipeline([
                ('svd', TruncatedSVD(num_comp)),
                ('svd_norm', Normalizer(copy=False)),
                ('red_featunion', build_ensemble([build_classifier(name) for name in clfs]))
            ]))
        ]))
    )

    if clf is not None:
        pipe.steps.append(('ensemblifier', clf))

    return pipe


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
        ('col_extr', JsonFields(0, ['title', 'body', 'url'])),
        ('squash', Squash()),
        ('vec', tfv)
    ]

    # Reduce dimensions of tfidf
    if num_comp > 0:
        if reducer == 'svd':
            vec_pipe.append(('dim_red', TruncatedSVD(num_comp)))
        elif reducer == 'kbest':
            vec_pipe.append(('dim_red', SelectKBest(chi2, k=num_comp)))
        elif reducer == 'percentile':
            vec_pipe.append(('dim_red', SelectPercentile(f_classif, percentile=num_comp)))

        vec_pipe.append(('norm', Normalizer()))

    return Pipeline(vec_pipe)


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
    pipe = Pipeline(chain)
    pipe.name = name
    return pipe


def build_ensemble(model_list, estimator=None):
    ''' Build an ensemble as a FeatureUnion of ModelTransformers and a final estimator using their
        predictions as input. '''

    models = []
    for i, model in enumerate(model_list):
        models.append(('model_transform'+str(i), ModelTransformer(model)))

    if not estimator:
        return FeatureUnion(models)
    else:
        return Pipeline([
            ('features', FeatureUnion(models)),
            ('estimator', estimator)
            ])



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


def build_classifier(name):
    ''' Return a dictionary with clf name as key, and model plus grids as values. '''

    if name == 'logit':
        model = LogisticRegression(
            penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True,
            intercept_scaling=1.0, class_weight=None, random_state=None) # random_state = random_seed
        model.grid_s = {'logit__C' : (0.1, 1, 5, 10)}
        model.grid_b = {'logit__C' : [(1)]}
        #
    elif name == 'sgd':
        model = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=0.0)
        model.grid_s = {'sgd__l1_ratio': (0, 0.25, 0.5, 1), 'sgd__alpha': (0.0001, 0.001, 0.01)}
        model.grid_b = {'sgd__l1_ratio': (1), 'sgd__alpha': (0.001)}
        #
    elif name == 'svc':
        model = SVC(kernel='linear', probability=True)
        model.grid_s = {'svc__C': [0.1, 1, 10, 100]}
        model.grid_b = {'svc__C': [1]}
        #
    elif name == 'rf':
        model = RandomForestClassifier(n_estimators=500, max_features=15, max_depth=10, min_samples_split=3, n_jobs=4, verbose=0)
        model.grid_s = {'rf__max_depth': [5, 10, 15], 'rf__max_features': [0.1, 0.5, 1.0], 'rf__min_samples_split': [3,6]}
        model.grid_b = {'rf__max_depth': [10], 'rf__max_features': [0.1], 'rf__min_samples_split': [6]}
        #
    elif name == 'gbc':
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=15, min_samples_split=9, verbose=0, max_features=8)
        model.grid_s = {'gbc__max_depth': [5, 10, 15], 'gbc__max_features': [0.1, 0.5, 1.0], 'gbc__learning_rate': [0.01, 0.05, 0.1], 'gbc__subsample': [0.5, 0.75, 1]}
        model.grid_b = {'gbc__max_depth': [5], 'gbc__max_features': [0.1], 'gbc__learning_rate': [0.01], 'gbc__subsample': [0.5]}
        #
    else:
        model = KNeighborsClassifier(n_neighbors=5)
        model.grid_s = {'knn__n_neighbors': [5, 10, 25]}
        model.grid_b = {'knn__n_neighbors': [25]}

    model.grid_c = model.grid_s.copy()
    model.grid_c['union__boil_vec__text_vec__vec__ngram_range'] = [(1, 2), (1, 3)]
    model.name = name
    return model


def get_all_classifiers():
    ''' Return a list of all classifiers. '''
    names = ['sgd', 'gbc', 'rf'] #'logit', 'sgd', 'svc'] #'knn']
    return [build_classifier(name) for name in names]


def build_all_pipes(vec_only, num_alch_cat, red_comp, reducer, stem):
    ''' Combine classifiers with transform pipes. '''
    clfs = get_all_classifiers()
    return [get_estimator_pipe(clf.name, clf, vec_only, num_alch_cat, red_comp, reducer, stem) for clf in clfs]


def build_simple_pipes():
    ''' Create classifier-only pipes (without prior transforms, if this is done upfront e.g.). '''
    clfs = get_all_classifiers()
    pipes = []
    for clf in clfs:
        pipe = Pipeline([(clf.name, clf)])
        pipe.name = clf.name
        pipes.append(pipe)
    return pipes


def grid_search(trf_in_fold=False, grid='grid_s', num_folds=5, reducer='svd', red_comp=0):
    """ Return best estimator from parameter grid. """

    random_seed = 40
    num_jobs = 4
    scoreFn = 'roc_auc'
    stem = 0

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
        pipes = build_simple_pipes()
    else:
        print "Building in-fold transforming pipes..."
        pipes = build_all_pipes(VEC_ONLY, num_alch_cat, red_comp, reducer, stem)

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
        this_grid = getattr(pipe.steps[-1][1], grid)
        gs = GridSearchCV(pipe, this_grid, scoring=scoreFn, cv=num_folds, n_jobs=num_jobs, verbose=2)

        with Timer(pipe.name + " fitting"):
            model = gs.fit(X_train, y_train)

        print "Best score on training set (CV): %0.3f" % gs.best_score_
        print "Best parameters set:"
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(this_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        for params, mean_score, scores in gs.grid_scores_:
            print "%0.4f (+/-%f) for %r: %r" % (mean_score, scores.std() / 2, params, scores)

        # Assess and predict (validation error etc.)
        print "\nPredict: "
        yp_val = model.predict(X_val)
        diff = y_val - yp_val
        print "Auroc val:", auroc(y_val, yp_val)
        print classification_report(y_val, yp_val)
        metrics.confusion_matrix

        best.append(gs.best_estimator_)
        print "10-fold cross-validation of best instance on whole set..."
        tr_roc = np.mean(cross_validation.cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='roc_auc', verbose=2))
        best_score.append(tr_roc)
        print "Mean score: ",  tr_roc
        print "================================================================================"

    return best, best_score


def class_report(conf_mat):
    tp, fp, fn, tn = conf_mat.flatten()
    measures = {}
    measures['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    measures['specificity'] = tn / (tn + fp)        # (true negative rate)
    measures['sensitivity'] = tp / (tp + fn)        # (recall, true positive rate)
    measures['precision'] = tp / (tp + fp)
    measures['f1score'] = 2*tp / (2*tp + fp + fn)
    return measures


def analyze_model(model=None, folds=10):
    ''' Run x-validation and return scores, averaged confusion matrix, and df with false positives and negatives '''

    grid = 'grid_b'
    scorer = 'roc_auc'

    X, y, X_test = load()
    y = y.values   # to numpy
    X = X.values
    if not model:
        model = load_model()

    # Manual x-validation to accumulate actual
    cv_skf = StratifiedKFold(y, n_folds=folds, shuffle=False, random_state=42)
    scores = []
    conf_mat = np.zeros((2, 2))      # Binary classification
    false_pos = Set()
    false_neg = Set()

    for train_i, val_i in cv_skf:
        X_train, X_val = X[train_i], X[val_i]
        y_train, y_val = y[train_i], y[val_i]

        print "Fitting fold..."
        model.fit(X_train, y_train) # Model pipeline first transforms data, then fits using final classifier

        print "Predicting fold..."
        y_pprobs = model.predict_proba(X_val)
        y_plabs = np.squeeze(model.predict(X_val))

        scores.append(roc_auc_score(y_val, y_pprobs[:, 1]))
        confusion = confusion_matrix(y_val, y_plabs)
        conf_mat += confusion

        # Collect indices of false positive and negatives
        fp_i = np.where((y_plabs==1) & (y_val==0))[0]
        fn_i = np.where((y_plabs==0) & (y_val==1))[0]
        false_pos.update(val_i[fp_i])
        false_neg.update(val_i[fn_i])

        print "Fold score: ", scores[-1]
        print "Fold CM: \n", confusion

    print "\nMean score: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores) * 2)
    conf_mat /= folds
    print "Mean CM: \n", conf_mat
    print "\nMean classification measures: \n"
    pprint(class_report(conf_mat))
    return scores, conf_mat, {'fp': sorted(false_pos), 'fn': sorted(false_neg)}



def train_custom(comp=120, stem=0, clf=None, xval=False):
    ''' Train best or ensemble of best from grid search. '''

    num_jobs = 5
    grid = 'grid_b'
    scorer = 'roc_auc'
    folds = 10

    X, y, X_test = load()

    model = get_custom_pipe(comp, stem, clf)

    if xval:
        print "Running cross-validation of model on whole set: "
        scores = cross_validation.cross_val_score(model, X, y, cv=folds, scoring=scorer, verbose=2)
        print("Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    # Train final model on whole dataset
    print "Fitting model to whole training set for prediction:"
    test_urlids = pd.read_table(testfnm, na_values=["?"])['urlid']
    model.fit(X, y)
    yp = model.predict_proba(X_test)[:,1]
    df = pd.DataFrame(data=test_urlids, columns=["urlid"])
    df['label'] = yp
    df.to_csv(datadir + 'submission.csv', index=False)
    print "Done. Submission file written to data dir."
    return model


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
        pipes = build_simple_pipes()
    else:
        print "Building in-fold transforming pipes..."
        pipes = build_all_pipes(VEC_ONLY, num_alch_cat, red_comp, reducer, stem)

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
    return model


def store_model(model):
    joblib.dump(model, datadir + 'model/model.pkl')


def load_model():
    return joblib.load(datadir + 'model/model.pkl')


if __name__ == "__main__":
    train()
