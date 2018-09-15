from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class RatioFeature(AbstractFeature):
    def __init__(self, numerator_feature_name, denominator_feature_name):
        self.numerator_feature_name = numerator_feature_name
        self.denominator_feature_name = denominator_feature_name
        super().__init__("{}_{}_ratio".format(numerator_feature_name, denominator_feature_name))

    def _do_transform(self, X):
        return X[self.numerator_feature_name] / X[self.denominator_feature_name]

