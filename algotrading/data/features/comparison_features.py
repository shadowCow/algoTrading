from abc import abstractmethod

from algotrading.data.features.feature_helpers import AbstractFeature
from algotrading.data.price_data_schema import price_data_schema as schema


class ComparisonFeature(AbstractFeature):
    def __init__(self, source_feature_name, comparison_op, target_value):
        self.source_feature_name = source_feature_name
        self.comparison_op = comparison_op
        self.target_value = target_value
        super().__init__("{}_{}_{}".format(source_feature_name, comparison_op, target_value))

    def _do_transform(self, X):
        return self._do_comparison_transform(X[self.source_feature_name])

    @abstractmethod
    def _do_comparison_transform(self, X):
        pass


class EqualFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "eq", target_value)

    def _do_comparison_transform(self, X):
        return X.eq(self.target_value)


class NotEqualFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "ne", target_value)

    def _do_comparison_transform(self, X):
        return X.ne(self.target_value)


class LessThanFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "lt", target_value)

    def _do_comparison_transform(self, X):
        return X.lt(self.target_value)


class LessOrEqualFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "le", target_value)

    def _do_comparison_transform(self, X):
        return X.le(self.target_value)


class GreaterThanFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "gt", target_value)

    def _do_comparison_transform(self, X):
        return X.gt(self.target_value)        


class GreaterOrEqualFeature(ComparisonFeature):
    def __init__(self, source_feature_name, target_value):
        super().__init__(source_feature_name, "ge", target_value)

    def _do_comparison_transform(self, X):
        return X.ge(self.target_value)

