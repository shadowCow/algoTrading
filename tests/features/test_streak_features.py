import unittest
import pandas as pd
import numpy as np

from tests.context import algotrading
from tests.context import dates
from tests.context import get_test_market_a
from tests.context import get_test_market_b
from tests.context import assert_elements_equal

import algotrading.data.features.intra_bar_features as ibf
import algotrading.data.features.streak_features as sf
from algotrading.data.market_direction import market_direction as md


class TestStreakFeatures(unittest.TestCase):

    def test_streak(self):
        market = get_test_market_a()
        direction = ibf.OpenCloseDirection()
        streak = sf.StreakCounterFeature(direction.name)

        with_direction = direction.fit_transform(market.data)
        
        expected_data = pd.Series([0,1,2,0,1], index=dates)
        transformed_data = streak.fit_transform(with_direction)[streak.name]

        assert_elements_equal(self, expected_data, transformed_data)

    def test_streak_counter_by_value(self):
        market = get_test_market_a()
        direction = ibf.OpenCloseDirection()
        streak = sf.StreakCounterFeature(direction.name)
        streak_counter_by_value = sf.StreakCounterByValueFeature(direction.name, md.up)

        with_direction = direction.fit_transform(market.data)
        with_streak = streak.fit_transform(with_direction)

        expected_data = pd.Series([0,1,2,0,0], index=dates)
        transformed_data = streak_counter_by_value.fit_transform(with_streak)[streak_counter_by_value.name]
        
        assert_elements_equal(self, expected_data, transformed_data)

