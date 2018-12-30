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


class IsUpBody(AbstractFeature):
    def __init__(self):
        super().__init__("is_up_body")

    def _do_transform(self, X):
        return (X[schema.close] > X[schema.open]).astype(int)


class IsDownBody(AbstractFeature):
    def __init__(self):
        super().__init__("is_down_body")

    def _do_transform(self, X):
        return (X[schema.close] < X[schema.open]).astype(int)


class BodyProportion(AbstractFeature):
    def __init__(self):
        super().__init__("body_proportion")

    def _do_transform(self, X):
        
        def the_lambda(x):
            body_size = abs(x[schema.close] - x[schema.open])
            bar_size = abs(x[schema.high] - x[schema.low])

            if (bar_size == 0.0):
                return 1
            else:
                return body_size / bar_size

        return X.apply(the_lambda, axis=1)
        
