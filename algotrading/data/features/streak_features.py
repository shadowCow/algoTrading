import pandas as pd
import numpy as np

from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class StreakCounterFeature(AbstractFeature):
    def __init__(self, source_feature_name):
        self.source_feature_name = source_feature_name
        super().__init__("{}_streak".format(source_feature_name))

    def _do_transform(self, X):
        # For an input column like:
        # True,True,True,False,True,False,False,False,False,True
        # this will get streak counts like:
        # 0,1,2,0,0,0,1,2,3,0
        return X.groupby(
          (X[self.source_feature_name] != X[self.source_feature_name].shift(1)).cumsum()
        ).cumcount()


class StreakCounterByValueFeature(AbstractFeature):
    def __init__(self, source_feature_name, target_value):
        self.source_feature_name = source_feature_name
        self.target_value = target_value
        self.streak_feature_name = "{}_streak".format(source_feature_name)
        super().__init__("{}_{}_streak".format(source_feature_name, target_value))

    def _do_transform(self, X):
        num_rows, unused_num_columns = X.shape
        new_series = pd.Series(np.zeros(num_rows), index=X.index.values)

        new_series[X[self.source_feature_name] == self.target_value] = X[self.streak_feature_name]
        return new_series

