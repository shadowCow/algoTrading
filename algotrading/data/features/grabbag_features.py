from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class MultiBarOCChange(AbstractFeature):
    def __init__(self, window_length):
        self.window_length = window_length
        super().__init__("multi_bar_oc_change_{}".format(window_length))

    def _do_transform(self, X):
        final_close = X[schema.close].rolling(self.window_length).apply(lambda s: s[-1])
        first_open = X[schema.open].rolling(self.window_length).apply(lambda s: s[0])

        return final_close - first_open


class SuddenVolatilityChange(AbstractFeature):
    def __init__(self, window_length):
        self.window_length = window_length
        super().__init__("sudden_volatility_change_{}".format(window_length))

    def _do_transform(self, X):
        hl_range = X[schema.high] - X[schema.low]
        max_range = hl_range.shift(1).rolling(self.window_length-1).max()

        return hl_range / max_range


class IsVolatilityExpansion(AbstractFeature):
    def __init__(self, window_length, multiplier=1.0):
        self.window_length = window_length
        self.multiplier = multiplier
        super().__init__("is_volatility_expansion_{}_{}".format(window_length, multiplier))

    def _do_transform(self, X):
        ve_name = "sudden_volatility_change_{}".format(self.window_length)
        return X[ve_name].gt(self.multiplier).astype(int)