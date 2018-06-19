import pandas as pd
from enum import Enum


class Feature(object):
    def __init__(self, name, variable_type, transform):
        self.name = name
        self.variable_type = variable_type
        self.transform = transform


def apply_features_to_markets(features, markets_data):
    def apply_features_to_market(features, market):
        # get the new data series by computing the features.
        features_series = list(map(lambda f: f.transform(market['data']), features))
        feature_names = [f.name for f in features]
        features_df = pd.concat(features_series, axis=1, keys=feature_names)
        final_df = pd.concat([market['data'], features_df], axis=1)
        return {
            "market": market,
            "data": final_df,
        }

    return list(map(lambda m: apply_features_to_market(features, m), markets_data))


class VariableTypes(Enum):
    continuous = 1
    binary = 2
    multiclass = 3
