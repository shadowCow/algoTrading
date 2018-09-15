from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class LeadFeature(AbstractFeature):
    def __init__(self, source_feature_name, distance):
        self.source_feature_name = source_feature_name
        self.distance = distance
        super().__init__("{}_lead_{}".format(source_feature_name, distance))

    def _do_transform(self, X):
        return X[self.source_feature_name].shift(-1*self.distance)


class LagFeature(AbstractFeature):
    def __init__(self, source_feature_name, distance):
        self.source_feature_name = source_feature_name
        self.distance = distance
        super().__init__("{}_lag_{}".format(source_feature_name, distance))

    def _do_transform(self, X):
        return X[self.source_feature_name].shift(self.distance)

        
        