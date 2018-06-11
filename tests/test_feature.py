import unittest
import pandas

from .context import algotrading
from .context import dates
from .context import get_test_data_frame_one
from .context import get_test_data_frame_two
from .context import get_test_feature_one
from .context import get_test_feature_two

import algotrading.data.feature as feature_engineering

class FeatureTests(unittest.TestCase):

    def test_apply_features_to_markets(self):
        markets = [
            {"market":{"symbol":"A"}, "data":get_test_data_frame_one()},
            {"market":{"symbol":"B"}, "data":get_test_data_frame_two()}
        ]

        features = [
            get_test_feature_one(),
            get_test_feature_two()
        ]

        markets_with_features = feature_engineering.apply_features_to_markets(
            features,
            markets
        )

        expected_data_one_f1 = pandas.Series([0.8, 1.4, 2.0, 1.4, 1.3], index=dates)
        actual_data_one_f1 = markets_with_features[0]['data']["feature_one"]

        expected_data_one_f2 = [True, True, True, False, False]
        actual_data_one_f2 = markets_with_features[0]['data']["feature_two"]

        expected_data_two_f1 = pandas.Series([0.65, 1.8, 1.15, 3.8, 1.55], index=dates)
        actual_data_two_f1 = markets_with_features[1]['data']["feature_one"]

        expected_data_two_f2 = [False, True, False, False, False]
        actual_data_two_f2 = markets_with_features[1]['data']["feature_two"]

        self.assertAlmostEqual(expected_data_one_f1[0], actual_data_one_f1[0])
        self.assertAlmostEqual(expected_data_one_f1[1], actual_data_one_f1[1])
        self.assertAlmostEqual(expected_data_one_f1[2], actual_data_one_f1[2])
        self.assertAlmostEqual(expected_data_one_f1[3], actual_data_one_f1[3])
        self.assertAlmostEqual(expected_data_one_f1[4], actual_data_one_f1[4])

        self.assertAlmostEqual(expected_data_one_f2[0], actual_data_one_f2[0])
        self.assertAlmostEqual(expected_data_one_f2[1], actual_data_one_f2[1])
        self.assertAlmostEqual(expected_data_one_f2[2], actual_data_one_f2[2])
        self.assertAlmostEqual(expected_data_one_f2[3], actual_data_one_f2[3])
        self.assertAlmostEqual(expected_data_one_f2[4], actual_data_one_f2[4])

        self.assertAlmostEqual(expected_data_two_f1[0], actual_data_two_f1[0])
        self.assertAlmostEqual(expected_data_two_f1[1], actual_data_two_f1[1])
        self.assertAlmostEqual(expected_data_two_f1[2], actual_data_two_f1[2])
        self.assertAlmostEqual(expected_data_two_f1[3], actual_data_two_f1[3])
        self.assertAlmostEqual(expected_data_two_f1[4], actual_data_two_f1[4])

        self.assertAlmostEqual(expected_data_two_f2[0], actual_data_two_f2[0])
        self.assertAlmostEqual(expected_data_two_f2[1], actual_data_two_f2[1])
        self.assertAlmostEqual(expected_data_two_f2[2], actual_data_two_f2[2])
        self.assertAlmostEqual(expected_data_two_f2[3], actual_data_two_f2[3])
        self.assertAlmostEqual(expected_data_two_f2[4], actual_data_two_f2[4])


if __name__ == '__main__':
    unittest.main()
