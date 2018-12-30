
from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class PivotHigh(AbstractFeature):
  def __init__(self, left_length, right_length):
    self.left_length = left_length
    self.right_length = right_length
    super().__init__("pivot_high_{}_{}".format(left_length, right_length))

  def _do_transform(self, X):
    # need high/low of left_length.
    # then need to shift it and see if the current
    # high low of right_length is the same.
    left_high = X[schema.high].rolling(
      window=self.left_length,
      min_periods=self.left_length
    ).max().shift(self.right_length)
    
    right_high = X[schema.high].rolling(
      window=self.right_length,
      min_periods=self.right_length
    ).max()


class PivotLow(AbstractFeature):
  def __init__(self, left_length, right_length):
    self.left_length = left_length
    self.right_length = right_length
    super().__init__("pivot_low_{}_{}".format(left_length, right_length))

  def _do_transform(self, X):
    left_low = X[schema.low].rolling(
      window=self.left_length,
      min_periods=self.left_length
    ).min().shift(self.right_length)

    right_low = X[schema.low].rolling(
      window=self.right_length,
      min_periods=self.right_length
    ).min()