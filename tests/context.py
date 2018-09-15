import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import pandas
import numpy
import algotrading
import algotrading.data.feature as feature_engineering
from algotrading.data.markets import Market, MarketWithData

dates = pandas.date_range(20010101, periods=5)

def get_test_data_frame_one():
    return pandas.DataFrame(
        numpy.array([
            [1.1, 1.9, 1.1, 1.9],
            [2.3, 3.5, 2.1, 3.3],
            [3.4, 5.0, 3.0, 4.7],
            [3.4, 3.6, 2.2, 2.4],
            [2.0, 2.0, 0.7, 0.8]
        ]),
        index = dates,
        columns = ['o','h','l','c']
    )

def get_test_data_frame_two():
    return pandas.DataFrame(
        numpy.array([
            [20.25, 20.40, 19.75, 19.90],
            [19.50, 21.30, 19.50, 21.15],
            [21.05, 21.15, 20.00, 20.10],
            [20.10, 23.65, 19.85, 20.10],
            [19.05, 19.20, 17.65, 18.00]
        ]),
        index = dates,
        columns = ['o','h','l','c']
    )

def get_test_market_a():
    return MarketWithData(Market("A", "Just_A"), get_test_data_frame_one())

def get_test_market_b():
    return MarketWithData(Market("B", "Just_B"), get_test_data_frame_two())

def get_test_feature_one():
    def transform(df):
        return df.h - df.l

    return feature_engineering.Feature('feature_one', 'continuous', transform)

def get_test_feature_two():
    def transform(df):
        return df.c > df.o

    return feature_engineering.Feature('feature_two', 'binary', transform)

def dummy_trade_simulator(df, feature_columns):
    # simple dummy trading model
    # buy on every even index
    df = df.assign(trading_action=numpy.array([1,0,1,0,1]))
    df = df.assign(outcome=(df.c - df.o) * df.trading_action)
    # hold from open to close
    return df


# helper function to do element by element equality of two series
def assert_elements_equal(test_object, expected_series, actual_series):
    test_object.assertEqual(expected_series.size, actual_series.size)

    if (expected_series.dtype == 'float64' and actual_series.dtype == 'float64'):
        assert_float_series_elements_equal(test_object, expected_series, actual_series)
    elif (expected_series.dtype == 'object' and actual_series.dtype == 'object'):
        assert_object_series_elements_equal(test_object, expected_series, actual_series)


def assert_float_series_elements_equal(test_object, expected_series, actual_series):
    for i in range(expected_series.size):
        both_nan = math.isnan(expected_series[i]) and math.isnan(actual_series[i])
        if both_nan:
            # They are considered equal for this test.
            pass
        else:
            test_object.assertAlmostEqual(expected_series[i], actual_series[i], places=3)


def assert_object_series_elements_equal(test_object, expected_series, actual_series):
    for i in range(expected_series.size):
        test_object.assertEqual(expected_series[i], actual_series[i])
