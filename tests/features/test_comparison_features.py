import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
import algotrading.data.features.comparison_features as cf
from algotrading.data.market_direction import market_direction as md
from algotrading.data.price_data_schema import price_data_schema as schema

class TestComparisonFeatures(unittest.TestCase):

    def test_equal(self):
        market = get_test_market_a()
        direction = ibf.OpenCloseDirection()
        equal = cf.EqualFeature(direction.name, md.down)

        with_direction = direction.fit_transform(market.data)

        expected_data = pd.Series([False, False, False, True, True], index=dates)
        transformed_data = equal.fit_transform(with_direction)[equal.name]
        
        assert_elements_equal(self, expected_data, transformed_data)

    def test_not_equal(self):
        market = get_test_market_a()
        direction = ibf.OpenCloseDirection()
        not_equal = cf.NotEqualFeature(direction.name, md.down)

        with_direction = direction.fit_transform(market.data)

        expected_data = pd.Series([True, True, True, False, False], index=dates)
        transformed_data = not_equal.fit_transform(with_direction)[not_equal.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_less_than(self):
        market = get_test_market_a()
        feature = cf.LessThanFeature(schema.low, 2.2)

        expected_data = pd.Series([True, True, False, False, True], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_less_or_equal(self):
        market = get_test_market_a()
        feature = cf.LessOrEqualFeature(schema.low, 2.2)

        expected_data = pd.Series([True, True, False, True, True], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_greater_than(self):
        market = get_test_market_a()
        feature = cf.GreaterThanFeature(schema.low, 2.2)

        expected_data = pd.Series([False, False, True, False, False], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_greater_or_equal(self):
        market = get_test_market_a()
        feature = cf.GreaterOrEqualFeature(schema.low, 2.2)

        expected_data = pd.Series([False, False, True, True, False], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

        