import pandas
from enum import Enum

class Feature:
    def __init__(self, name, variableType, transform):
        self.name = name
        self.variableType = variableType
        self.transform = transform


def apply_features_to_markets(features, markets_data):
    def apply_features_to_market(features, market):
        # get the new data series by computing the features.
        features_series = list(map(lambda f: f.transform(market['data']), features))
        feature_names = [f.name for f in features]
        features_df = pandas.concat(features_series, axis=1, keys=feature_names)
        final_df = pandas.concat([market['data'], features_df], axis=1)
        return {
            "market": market,
            "data": final_df
        }

    return list(map(lambda m: apply_features_to_market(features, m), markets_data))

class VariableTypes(Enum):
    continuous = 1
    binary = 2
    multiclass = 3
