import unittest
import pandas as pd
import numpy as np

from .context import algotrading
from .context import dates
from .context import get_test_data_frame_one
from .context import get_test_data_frame_two
from .context import get_test_market_a
from .context import get_test_market_b

import algotrading.data.feature as feature_engineering
import algotrading.data.features as features

class TestVariousFeatures(unittest.TestCase):

    def test_oc_change(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.oc_change
        )

        expected_data = pd.Series([0.8, 1.0, 1.3, -1.0, -1.2], index=dates)
        actual_data = markets_with_features[0]['data']['oc_change']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_hl_range(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.hl_range
        )

        expected_data = pd.Series([0.8, 1.4, 2.0, 1.4, 1.3], index=dates)
        actual_data = markets_with_features[0]['data']['hl_range']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_ol_range(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.ol_range
        )

        expected_data = pd.Series([0, 0.2, 0.4, 1.2, 1.3], index=dates)
        actual_data = markets_with_features[0]['data']['ol_range']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_oh_range(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.oh_range
        )

        expected_data = pd.Series([0.8, 1.2, 1.6, 0.2, 0], index=dates)
        actual_data = markets_with_features[0]['data']['oh_range']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_oc_is_up(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.oc_is_up
        )

        expected_data = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=dates)
        actual_data = markets_with_features[0]['data']['oc_is_up']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_oc_is_down(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.oc_is_down
        )

        expected_data = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0], index=dates)
        actual_data = markets_with_features[0]['data']['oc_is_down']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_oc_is_flat(self):
        markets_with_features = feature_engineering.apply_features_to_markets(
            [features.oc_is_flat],
            [get_test_market_b()]
        )

        expected_data = pd.Series([0.0, 0.0, 0.0, 1.0, 0.0], index=dates)
        actual_data = markets_with_features[0]['data']['oc_is_flat']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_moving_average(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.moving_average_feature(features.raw_data_as_feature('c'), 3)
        )

        expected_data = pd.Series([np.nan, np.nan, 3.3, 3.467, 2.633], index=dates)
        actual_data = markets_with_features[0]['data']['ma_3_c']

        self.assertTrue(pd.isnull(actual_data[0]))
        self.assertTrue(pd.isnull(actual_data[1]))
        self.assertAlmostEqual(expected_data[2], actual_data[2])
        self.assertAlmostEqual(expected_data[3], actual_data[3], places=3)
        self.assertAlmostEqual(expected_data[4], actual_data[4], places=3)

    def test_normalized(self):
        # TODO write test
        pass

    def test_streak_counter(self):
        markets_with_features = feature_engineering.apply_features_to_markets(
            [features.streak_counter_feature(features.oc_is_up)],
            [get_test_market_a()]
        )

        expected_data = pd.Series([0,1,2,0,-1], index=dates)
        actual_data = markets_with_features[0]['data']['streak_oc_is_up']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_rolling_max(self):
        markets_with_features = feature_engineering.apply_features_to_markets(
            [features.rolling_max_feature(features.raw_data_as_feature('h'), 3)],
            [get_test_market_b()]
        )

        expected_data = pd.Series([np.nan, np.nan, 21.30, 23.65, 23.65], index=dates)
        actual_data = markets_with_features[0]['data']['max_3_h']
        self.assertTrue(pd.isnull(actual_data[0]))
        self.assertTrue(pd.isnull(actual_data[1]))
        self.assertAlmostEqual(expected_data[2], actual_data[2])
        self.assertAlmostEqual(expected_data[3], actual_data[3], places=3)
        self.assertAlmostEqual(expected_data[4], actual_data[4], places=3)

    def test_min_over_window(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.rolling_min_feature(features.raw_data_as_feature('l'), 3)
        )

        expected_data = pd.Series([np.nan, np.nan, 1.1, 2.1, 0.7], index=dates)
        actual_data = markets_with_features[0]['data']['min_3_l']
        self.assertTrue(pd.isnull(actual_data[0]))
        self.assertTrue(pd.isnull(actual_data[1]))
        self.assertAlmostEqual(expected_data[2], actual_data[2])
        self.assertAlmostEqual(expected_data[3], actual_data[3], places=3)
        self.assertAlmostEqual(expected_data[4], actual_data[4], places=3)

    def test_is_max_over_window(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.is_max_feature(features.raw_data_as_feature('c'), 3)
        )

        # i think nan would be preferred for the first 2, but whatever.
        expected_data = pd.Series([False, False, True, False, False])
        actual_data = markets_with_features[0]['data']['is_max_3_c']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    def test_is_min_over_window(self):
        markets_with_features = TestVariousFeatures.get_test_data_with_feature(
            features.is_min_feature(features.raw_data_as_feature('c'), 3)
        )

        # i think nan would be preferred for the first 2, but whatever.
        expected_data = pd.Series([False, False, False, True, True])
        actual_data = markets_with_features[0]['data']['is_min_3_c']
        TestVariousFeatures.assert_elements_equal(self, expected_data, actual_data)

    # helper function to do element by element equality of two series
    @staticmethod
    def assert_elements_equal(test_object, expected_series, actual_series):
        for i in range(5):
            test_object.assertAlmostEqual(expected_series[i], actual_series[i], places=3)

    @staticmethod
    def get_test_data_with_feature(feature):
        return feature_engineering.apply_features_to_markets(
            [feature],
            [get_test_market_a()]
        )
