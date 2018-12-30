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

    def test_median(self):
        market = get_test_market_a()
        median = wf.Median(schema.close, 3)

        expected_data = pd.Series([np.nan, np.nan, 3.3, 3.3, 2.4], index=dates)
        transformed_data = median.fit_transform(market.data)[median.name]

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
        
    def test_up_bar_proportion(self):
        market = get_test_market_a()
        up_bar_proportion = wf.UpBarProportion(3)

        data_with_is_up_body = ibf.IsUpBody().fit_transform(market.data)
        
        expected_data = pd.Series([np.nan, np.nan, 1.0, 2/3, 1/3], index=dates)
        transformed_data = up_bar_proportion.fit_transform(data_with_is_up_body)[up_bar_proportion.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_down_bar_proportion(self):
        market = get_test_market_a()
        down_bar_proportion = wf.DownBarProportion(3)

        data_with_is_down_body = ibf.IsDownBody().fit_transform(market.data)

        expected_data = pd.Series([np.nan, np.nan, 0.0, 1/3, 2/3], index=dates)
        transformed_data = down_bar_proportion.fit_transform(data_with_is_down_body)[down_bar_proportion.name]

        assert_elements_equal(self, expected_data, transformed_data)
        