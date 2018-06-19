import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas
import numpy
import algotrading
import algotrading.data.feature as feature_engineering

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
    return {"market":{"symbol":"A"}, "data":get_test_data_frame_one()}

def get_test_market_b():
    return {"market":{"symbol":"B"}, "data":get_test_data_frame_two()}

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
