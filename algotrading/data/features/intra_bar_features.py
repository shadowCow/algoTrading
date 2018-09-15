from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema
from algotrading.data.market_direction import get_feature_direction


class OpenCloseRange(AbstractFeature):
    def __init__(self):
        super().__init__("oc_range")

    def _do_transform(self, X):
        return (X[schema.close] - X[schema.open]).abs()


class HighLowRange(AbstractFeature):
    def __init__(self):
        super().__init__("hl_range")

    def _do_transform(self, X):
        return X[schema.high] - X[schema.low]


class OpenLowRange(AbstractFeature):
    def __init__(self):
        super().__init__("ol_range")

    def _do_transform(self, X):
        return X[schema.open] - X[schema.low]


class OpenHighRange(AbstractFeature):
    def __init__(self):
        super().__init__("oh_range")

    def _do_transform(self, X):
        return X[schema.high] - X[schema.open]


class OpenCloseChange(AbstractFeature):
    def __init__(self):
        super().__init__("oc_change")

    def _do_transform(self, X):
        return X[schema.close] - X[schema.open]
        

class OpenCloseDirection(AbstractFeature):
    def __init__(self):
        super().__init__("oc_direction")

    def _do_transform(self, X):
        return get_feature_direction(X[schema.close] - X[schema.open])


