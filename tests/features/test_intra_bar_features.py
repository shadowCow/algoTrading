import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
from algotrading.data.market_direction import market_direction as md


class TestIntraBarFeatures(unittest.TestCase):

    def test_oc_range(self):
        market = get_test_market_a()
        feature = ibf.OpenCloseRange()

        expected_data = pd.Series([0.8, 1.0, 1.3, 1.0, 1.2], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_hl_range(self):
        market = get_test_market_a()
        feature = ibf.HighLowRange()

        expected_data = pd.Series([0.8, 1.4, 2.0, 1.4, 1.3], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_ol_range(self):
        market = get_test_market_a()
        feature = ibf.OpenLowRange()

        expected_data = pd.Series([0, 0.2, 0.4, 1.2, 1.3], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_oh_range(self):
        market = get_test_market_a()
        feature = ibf.OpenHighRange()

        expected_data = pd.Series([0.8, 1.2, 1.6, 0.2, 0], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_oc_change(self):
        market = get_test_market_a()
        feature = ibf.OpenCloseChange()
        
        expected_data = pd.Series([0.8, 1.0, 1.3, -1.0, -1.2], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_oc_direction(self):
        market = get_test_market_b()
        feature = ibf.OpenCloseDirection()

        expected_data = pd.Series([md.down, md.up, md.down, md.flat, md.down], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]
        
        assert_elements_equal(self, expected_data, transformed_data)

    def test_is_up_body(self):
        market = get_test_market_b()
        feature = ibf.IsUpBody()

        expected_data = pd.Series([0, 1, 0, 0, 0], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_is_down_body(self):
        market = get_test_market_b()
        feature = ibf.IsDownBody()

        expected_data = pd.Series([1,0,1,0,1], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_body_proportion(self):
        market = get_test_market_a()
        feature = ibf.BodyProportion()

        expected_data = pd.Series([1.0, 1.0/1.4, 1.3/2.0, 1.0/1.4, 1.2/1.3], index=dates)
        transformed_data = feature.fit_transform(market.data)[feature.name]

        assert_elements_equal(self, expected_data, transformed_data)