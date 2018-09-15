import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
import algotrading.data.features.shifted_features as shf
from algotrading.data.market_direction import market_direction as md
from algotrading.data.price_data_schema import price_data_schema as schema

class TestShiftedFeatures(unittest.TestCase):

    def test_lead(self):
        market = get_test_market_a()
        lead = shf.LeadFeature(schema.open, 1)

        expected_data = pd.Series([2.3, 3.4, 3.4, 2.0, np.nan], index=dates)
        transformed_data = lead.fit_transform(market.data)[lead.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_lag(self):
        market = get_test_market_a()
        lag = shf.LagFeature(schema.open, 1)

        expected_data = pd.Series([np.nan, 1.1, 2.3, 3.4, 3.4], index=dates)
        transformed_data = lag.fit_transform(market.data)[lag.name]

        assert_elements_equal(self, expected_data, transformed_data)

        