import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.preprocessing import binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve, validation_curve

from pprint import pprint

import train


#
def plot_learning_curve(model, X, y, scorer, sizes=np.linspace(0.1, 1, 5), cv=None, n_jobs=5, ylim=None, title="Xval. learning curve"):
    ''' Plot learning curve for model on data '''

    df = pd.DataFrame()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=sizes)
    df['sizes_p'] = sizes
    df['sizes_n'] = train_sizes
    df['train_mean'] = 1 - np.mean(train_scores, axis=1)
    df['train_std'] = np.std(train_scores, axis=1)
    df['test_mean'] = 1 - np.mean(test_scores, axis=1)
    df['test_std'] = np.std(test_scores, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Size of training set")
    ax.set_ylabel("Error (1-score)")
    ax.grid()
    ax.fill_between(sizes, df.train_mean - df.train_std, df.train_mean + df.train_std, alpha=0.1, color="r")
    ax.fill_between(sizes, df.test_mean - df.test_std, df.test_mean + df.test_std, alpha=0.1, color="g")
    ax.plot(sizes, df.train_mean, 'o-', color="r", label="Training")
    ax.plot(sizes, df.test_mean, 'o-', color="g", label="Test")
    ax.legend(loc="best")
    fig.show()
    return df, fig


def plot_validation_curve(model, X, y, scorer, param_name, param_range=np.linspace(0.1, 1, 5), cv=None, n_jobs=5,
    ylim=None, title="Xval. validation curve"):
    ''' Plot learning curve for model on data '''

    df = pd.DataFrame()
    df['param_range'] = param_range
    train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scorer, n_jobs=n_jobs)
    df['train_mean'] = 1 - np.mean(train_scores, axis=1)
    df['train_std'] = np.std(train_scores, axis=1)
    df['test_mean'] = 1 - np.mean(test_scores, axis=1)
    df['test_std'] = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Parameter value")
    plt.ylabel("Error (1-score)")
    plt.grid()
    plt.semilogx(param_range, df.train_mean, color="r", label="Training")
    plt.fill_between(param_range, df.train_mean - df.train_std, df.train_mean + df.train_std, alpha=0.1, color="r")
    plt.semilogx(param_range, df.test_mean, color="g", label="Test")
    plt.fill_between(param_range, df.test_mean - df.test_std, df.test_mean + df.test_std, alpha=0.1, color="g")
    plt.legend(loc="best")
    plt.show()
    return df, plt


def test_lc():
    digits = load_digits()
    X, y = digits.data, digits.target

    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10, test_size=0.2, random_state=0)
    model = GaussianNB()
    plot_learning_curve(model, X, y, scorer='None', cv=cv, n_jobs=2, ylim=(0.0, 0.5), title="Learning Curves (Naive Bayes)")

    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10, test_size=0.2, random_state=0)
    model = SVC(gamma=0.001)
    plot_learning_curve(model, X, y, scorer='None', cv=cv, n_jobs=2, ylim=(0.0, 0.5), title="Learning Curves (SVM)")
    plt.show()


def test_vc():
    digits = load_digits()
    X, y = digits.data, digits.target
    p_range = np.logspace(-6, -1, 5)
    cv = cross_validation.ShuffleSplit(digits.data.shape[0], n_iter=10, test_size=0.2, random_state=0)
    model = SVC()
    plot_validation_curve(model, X, y, scorer='accuracy', param_name="gamma", param_range=p_range,
        cv=cv, n_jobs=2, ylim=(0.0, 0.5), title="SVC validation curve ($\gamma$)")
    plt.show()


#
# Tf-idf analysis of results
#
#features = vec.get_feature_names()      # list of features
#vocab = vec.vocabulary_                 # dict of feature names (key) to index (in feature_names)
#weights = vec.idf_                      # Calculated values for each feature (during fit)

# sorted_ind = np.argsort(weights)[::-1]  # Descending ordered idf weights
# top_n_features = [(features[i], vec.idf_[i]) for i in sorted_ind[:top_n]]
# print "top", top_n, " idf-only features: \n"
# pprint(top_n_features)


def get_tfidf(stem=0):
    X, y, _ = train.load()
    vec_pipe = train.get_vec_pipe(num_comp=0, reducer='svd', stem=stem)
    Xtr = vec_pipe.fit_transform(X)
    vec = vec_pipe.named_steps['vec']
    return vec, Xtr, y


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in vec and return them with their corresponding feature names. '''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return top n features on average most important amongst docs in rows grp_ids '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    for i, df in enumerate(dfs):

        ax = fig.add_subplot(len(dfs), 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        if i == len(dfs)-1:
            ax.set_xlabel("Feature name", labelpad=14, fontsize=14)
        ax.set_ylabel("Tf-Idf score", labelpad=16, fontsize=14)
        #if i == 0:
        ax.set_title("Mean Tf-Idf scores for label = " + str(df.label), fontsize=16)

        x = range(1, len(df)+1)
        ax.bar(x, df.tfidf, align='center', color='#3F5D7D')
        #ax.lines[0].set_visible(False)
        ax.set_xticks(x)
        ax.set_xlim([0,len(df)+1])
        xticks = ax.set_xticklabels(df.feature)
        #plt.ylim(0, len(df)+2)
        plt.setp(xticks, rotation='vertical') #, ha='right', va='top')
        plt.subplots_adjust(bottom=0.24, right=1, top=0.97, hspace=0.9)

    plt.show()


def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):

        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)

    plt.show()


def test_tfidf(misclf=None):
    vec, X, y = get_tfidf()
    features = vec.get_feature_names()
    n = 25
    min_score = 0.1

    print "Overall top features: \n"
    overall = top_mean_feats(X, features, grp_ids=None, min_tfidf=min_score, top_n=n)
    pprint(overall)

    print "\nTop feature by class: \n"
    by_class = top_feats_by_class(X, y, features, min_tfidf=min_score, top_n=n)
    for df in by_class:
        print "\nClass ", df.label
        pprint(df)
    plot_tfidf_classfeats_h(by_class)

    if misclf:
        print "\nTop features in false positives: \n"
        fp = top_mean_feats(X, features, misclf['fp'], min_tfidf=min_score, top_n=n)
        pprint(fp)

        print "\nTop features in false negatives: \n"
        fn = top_mean_feats(X, features, misclf['fn'], min_tfidf=min_score, top_n=n)
        pprint(fn)



