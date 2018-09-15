from abc import abstractmethod

from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class WindowFeature(AbstractFeature):
    def __init__(self, source_feature_name, window_op_name, window_length):
        self.source_feature_name = source_feature_name
        self.window_op_name = window_op_name
        self.window_length = window_length
        super().__init__("{}_{}_{}".format(source_feature_name, window_op_name, window_length))

    def _do_transform(self, X):
        return self._do_transform_window(
          X[self.source_feature_name].rolling(
            window=self.window_length,
            min_periods=self.window_length
          )
        )

    @abstractmethod
    def _do_transform_window(self, X):
        pass


class Average(WindowFeature):
    def __init__(self, source_feature_name, window_length):
        super().__init__(source_feature_name, "average", window_length)

    def _do_transform_window(self, X):
        return X.mean()
        

class Max(WindowFeature):
    def __init__(self, source_feature_name, window_length):
        super().__init__(source_feature_name, "max", window_length)

    def _do_transform_window(self, X):
        return X.max()


class Min(WindowFeature):
    def __init__(self, source_feature_name, window_length):
        super().__init__(source_feature_name, "min", window_length)

    def _do_transform_window(self, X):
        return X.min()

      
class Change(WindowFeature):
    def __init__(self, source_feature_name, window_length):
        super().__init__(source_feature_name, "change", window_length)

    def _do_transform_window(self, X):
        return X.apply(lambda s: s[-1] - s[0])


class Range(WindowFeature):
    def __init__(self, source_feature_name, window_length):
        super().__init__(source_feature_name, "range", window_length)

    def _do_transform_window(self, X):
        return X.max() - X.min()

