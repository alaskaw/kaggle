# Custom transformers idea: http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import binarize
from urlparse import urlparse

# Helpers
# ------------------------------------------------------------------------------------------------
def unsquash(X):
    ''' Transform vector of dim (n,) into (n,1) '''
    if len(X.shape) == 1 or X.shape[0] == 1:
        return np.asarray(X).reshape((len(X), 1))
    else:
        return X


def squash(X):
    ''' Transform vector of dim (n,1) into (n,) '''
    return np.squeeze(np.asarray(X))


def extract_json(txt, fields):
    ''' Extract specified fields from txt '''
    if not isinstance(fields, list):
        fields = [fields]
    obj = json.loads(txt)
    res = " ".join([obj.get(field) for field in fields if obj.get(field)])
    return res


def extract_url(url, field):
    ''' Extract given url fields from url '''
    parsed = urlparse(url)
    if field == 'netloc':
        return parsed.netloc
    elif field == 'scheme':
        return parsed.scheme
    else:
        return parsed.path



# Transformers
# ------------------------------------------------------------------------------------------------
class Transformer(TransformerMixin):
    ''' Base class for pure transformers that don't need a fit method '''

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X

    def get_params(self, deep=True):
        return dict()


# Wraps other models to perform transformation, e.g. a kmeans to obtain cluster index as new feature
# Or for use in a model stack (an estimator trained using as features the predictions of other models)
class ModelTransformer(TransformerMixin):
    ''' Use model predictions as transformer '''
    def __init__(self, model, probs=True):
        self.model = model
        self.probs = probs

    def get_params(self, deep=True):
        return dict(model=self.model, probs=self.probs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        if self.probs:
            Xtrf = self.model.predict_proba(X)[:, 1]
        else:
            Xtrf = self.model.predict(X)
        return unsquash(Xtrf)


class FeatureStack(BaseEstimator, TransformerMixin):
    '''Stacks several transformer objects to yield concatenated features. Similar to FeatureUnion,
    a list of tuples ``(name, estimator)`` is passed to the constructor. Not parallel. But
    useful for debugging when e.g. FeatureUnion doesn't work
    '''
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        #print "\nStack: starting: " + str(self.transformer_list)
        features = []
        for name, trans in self.transformer_list:
            print "Stack next step: ", name
            Xtr = trans.transform(X)
            features.append(Xtr)
            print "Stack step ended: ", name, " ", type(Xtr), " ", Xtr.shape
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            # Convert to sparse if necessary, otherwise cannot be hstack'ed
            features = [sparse.csr_matrix(unsquash(f)) for f in features]
            features = sparse.hstack(features).tocsr()
        else:
            features = np.column_stack(features)

        print "Stack: finished. Shape: ", features.shape, "\n"
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStack, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out


class EnsembleBinaryClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    ''' Average or majority-vote several different classifiers. Assumes input is a matrix of individual predictions,
        such as the output of a FeatureUnion of ModelTransformers [n_samples, n=predictors].
        Also see http://sebastianraschka.com/Articles/2014_ensemble_classifier.html.'''

    def __init__(self, mode, weights=None):
        self.mode = mode
        self.weights = weights

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        ''' Predict (weighted) probabilities '''
        probs = np.average(X, axis=1, weights=self.weights)
        return np.column_stack((1-probs, probs))

    def predict(self, X):
        ''' Predict class labels. '''
        if self.mode == 'average':
            return binarize(self.predict_proba(X)[:,[1]], 0.5)
        else:
            res = binarize(X, 0.5)
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int), self.weights).argmax(), axis=1, arr=res)



class Select(Transformer):
    '''  Extract specified columns from a pandas df or numpy array '''

    def __init__(self, columns=0, to_np=True):
        self.columns = columns
        self.to_np = to_np

    def get_params(self, deep=True):
        return dict(columns=self.columns, to_np=self.to_np)

    def transform(self, X, **transform_params):
        if isinstance(X, pd.DataFrame):
            allint = isinstance(self.columns, int) or (isinstance(self.columns, list) and all([isinstance(x, int) for x in self.columns]))
            if allint:
                res = X.ix[:, self.columns]
            elif all([isinstance(x, str) for x in self.columns]):
                res = X[self.columns]
            else:
                print "Select error: mixed or wrong column type."
                res = X

            if self.to_np:
                res = unsquash(res.values)
        else:
            res = unsquash(X[:, self.columns])

        return res



class JsonFields(Transformer):
    ''' Extract json encoded fields from a numpy array. Returns (iterable) numpy array so it can be used
        as input to e.g. Tdidf '''

    def __init__(self, column, fields=[], join=True):
        self.column = column
        self.fields = fields
        self.join = join

    def get_params(self, deep=True):
        return dict(column=self.column, fields=self.fields, join=self.join)

    def transform(self, X, **transform_params):
        col = Select(self.column, to_np=True).transform(X)
        res = np.vectorize(extract_json, excluded=['fields'])(col, fields=self.fields)
        return res


# Converts url strings into parts of the url (such as base domain)
class UrlField(Transformer):
    def __init__(self, column, field='netloc'):
        self.column = column
        self.field = field

    def get_params(self, deep=True):
        return dict(column=self.column, field=self.field)

    def transform(self, X, **transform_params):
        col = Select(self.column, to_np=True).transform(X)
        res = np.vectorize(extract_url)(col, field=self.field)
        return res


# Assumes column of data type compatible with len function
class Length(Transformer):
    def __init__(self, column=0):
        self.column = column

    def get_params(self, deep=True):
        return dict(column=self.column)

    def transform(self, X, **transform_params):
        col = Select(self.column, to_np=True).transform(X)
        res = np.vectorize(len)(col)
        res = res.astype(float)
        return unsquash(res)


# Turns a single array into a matrix with single column
# This allows concatenating the predictions of different estimators to be used as
# features in a feature union (since hstack doesn't produce a 2-ol matrix from two arrays).
# Note: this is already been taken care of in ModelTransformer
class Squash(Transformer):
    def transform(self, X, **transform_params):
        return squash(X)

class Unsquash(Transformer):
    def transform(self, X, **transform_params):
        return unsquash(X)

class Float(Transformer):
    def transform(self, X, **transform_params):
        return X.astype(float)


# General info about dataset
class DatasetReporter(Transformer):
    def transform(self, X, **transform_params):
        print "Data set: type = ", type(X),
        print " shape = ", X.shape
        return X


