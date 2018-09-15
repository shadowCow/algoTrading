import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
import algotrading.data.features.window_features as wf
from algotrading.data.market_direction import market_direction as md
from algotrading.data.price_data_schema import price_data_schema as schema


class TestWindowFeatures(unittest.TestCase):

    def test_moving_average(self):
        market = get_test_market_a()
        average = wf.Average(schema.close, 3)

        expected_data = pd.Series([np.nan, np.nan, 3.3, 3.467, 2.633], index=dates)
        transformed_data = average.fit_transform(market.data)[average.name]        

        assert_elements_equal(self, expected_data, transformed_data)

    def test_rolling_max(self):
        market = get_test_market_b()
        max = wf.Max(schema.high, 3)

        expected_data = pd.Series([np.nan, np.nan, 21.30, 23.65, 23.65], index=dates)
        transformed_data = max.fit_transform(market.data)[max.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_rolling_min(self):
        market = get_test_market_b()
        min = wf.Min(schema.low, 3)

        expected_data = pd.Series([np.nan, np.nan, 19.50, 19.50, 17.65], index=dates)
        transformed_data = min.fit_transform(market.data)[min.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_change(self):
        market = get_test_market_a()
        change = wf.Change(schema.close, 3)

        expected_data = pd.Series([np.nan, np.nan, 2.8, -0.9, -3.9], index=dates)
        transformed_data = change.fit_transform(market.data)[change.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_range(self):
        market = get_test_market_a()
        range = wf.Range(schema.high, 3)

        expected_data = pd.Series([np.nan, np.nan, 3.1, 1.5, 3.0], index=dates)
        transformed_data = range.fit_transform(market.data)[range.name]

        assert_elements_equal(self, expected_data, transformed_data)
        
