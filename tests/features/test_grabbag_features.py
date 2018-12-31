import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

from algotrading.data.features import grabbag_features as gbf


class TestGrabbagFeatures(unittest.TestCase):

    def test_multibar_oc_change(self):
        market = get_test_market_a()
        multi_bar_oc_change = gbf.MultiBarOCChange(3)

        expected_data = pd.Series([np.nan, np.nan, 3.6, 0.1, -2.6], index=dates)
        transformed_data = multi_bar_oc_change.fit_transform(market.data)[multi_bar_oc_change.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_sudden_volatility_change(self):
        market = get_test_market_a()
        sudden_volatility_change = gbf.SuddenVolatilityChange(3)
      
        expected_data = pd.Series([np.nan, np.nan, 2.0/1.4, 1.4/2.0, 1.3/2.0], index=dates)
        transformed_data = sudden_volatility_change.fit_transform(market.data)[sudden_volatility_change.name]
        
        assert_elements_equal(self, expected_data, transformed_data)

    def test_is_volatility_expansion(self):
        market = get_test_market_a()
        is_volatility_expansion = gbf.IsVolatilityExpansion(3, 1.0)
        with_sudden_volatility_change = gbf.SuddenVolatilityChange(3).fit_transform(market.data)
        
        expected_data = pd.Series([0, 0, 1, 0, 0], index=dates, dtype="int64")
        transformed_data = is_volatility_expansion.fit_transform(with_sudden_volatility_change)[is_volatility_expansion.name]
        
        assert_elements_equal(self, expected_data, transformed_data)