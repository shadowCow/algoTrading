from abc import ABC, abstractmethod
from enum import Enum

from sklearn.base import BaseEstimator, TransformerMixin


class AbstractFeature(ABC, BaseEstimator, TransformerMixin):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.name] = self._do_transform(X)
        return X

    @abstractmethod
    def _do_transform(self, X):
        pass


class VariableTypes(Enum):
    continuous = 1
    discrete = 2
    binary = 3
    multiclass = 4


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values