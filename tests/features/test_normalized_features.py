import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
import algotrading.data.features.normalized_features as nf
from algotrading.data.market_direction import market_direction as md
from algotrading.data.price_data_schema import price_data_schema as schema

class TestNormalizedFeatures(unittest.TestCase):

    def test_ratio(self):
        market = get_test_market_a()
        feature = nf.RatioFeature(schema.high, schema.low)

        expected_data = pd.Series([1.727, 1.667, 1.667, 1.636, 2.857], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]
        
        assert_elements_equal(self, expected_data, transformed_data)

