from algotrading.data.feature import Feature, VariableTypes
import pandas as pd
import numpy as np

def raw_data_as_feature(col_name):
    return Feature(
        col_name,
        VariableTypes.continuous,
        lambda df: df[col_name]
    )

# intra-pricebar data point differences.
oc_change = Feature(
    'oc_change',
    VariableTypes.continuous,
    lambda df: df.c - df.o
)

hl_range = Feature(
    'hl_range',
    VariableTypes.continuous,
    lambda df: df.h - df.l
)

ol_range = Feature(
    'ol_range',
    VariableTypes.continuous,
    lambda df: (df.o - df.l).abs()
)

oh_range = Feature(
    'oh_range',
    VariableTypes.continuous,
    lambda df: (df.o - df.h).abs()
)

# intra-price bar directions
oc_is_up = Feature(
    'oc_is_up',
    VariableTypes.binary,
    lambda df: df.c > df.o
)

oc_is_down = Feature(
    'oc_is_down',
    VariableTypes.binary,
    lambda df: df.c < df.o
)

oc_is_flat = Feature(
    'oc_is_flat',
    VariableTypes.binary,
    lambda df: df.o == df.c
)

# window features
def moving_average_feature(feature, length):
    return Feature(
        'ma_{}_{}'.format(length, feature.name),
        VariableTypes.continuous,
        lambda df: moving_average(feature.transform(df), length)
    )

def rolling_max_feature(feature, length):
    return Feature(
        'max_{}_{}'.format(length, feature.name),
        feature.variable_type,
        lambda df: max_over_window(feature.transform(df), length)
    )

def rolling_min_feature(feature, length):
    return Feature(
        'min_{}_{}'.format(length, feature.name),
        feature.variable_type,
        lambda df: min_over_window(feature.transform(df), length)
    )

def streak_counter_feature(feature):
    def include_feature_in_df(df, feature):
        new_col = feature.transform(df).rename(feature.name)
        new_df = pd.concat([df, new_col], axis=1)
        return streak_counter(
            new_df,
            feature.name
        )

    return Feature(
        'streak_{}'.format(feature.name),
        VariableTypes.discrete,
        lambda df: include_feature_in_df(df, feature)
    )

def is_max_feature(feature, length):
    return Feature(
        'is_max_{}_{}'.format(length, feature.name),
        feature.variable_type,
        lambda df: is_max_over_window(feature.transform(df), length)
    )

def is_min_feature(feature, length):
    return Feature(
        'is_min_{}_{}'.format(length, feature.name),
        feature.variable_type,
        lambda df: is_min_over_window(feature.transform(df), length)
    )

# common transformations
def moving_average(column, length):
    return column.rolling(window=length, min_periods=length).mean()

def normalized(column, normalization):
    return column / normalization

def streak_counter(df, col_name):
    # For an input column like:
    # True,True,True,False,True,False,False,False,False,True
    # this will get streak counts like:
    # 0,1,2,0,0,0,1,2,3,0
    cum_counted = df.groupby((df[col_name] != df[col_name].shift(1)).cumsum()).cumcount()

    # we want to differentiate between streaks of true values and streaks of false values
    # so we use negative values for False streaks, like:
    # 0,1,2,0,0,0,-1,-2,-3,0
    cum_counted[df[col_name] == False] = cum_counted * -1
    return cum_counted

def max_over_window(column, length):
    return column.rolling(window=length, min_periods=length).max()

def min_over_window(column, length):
    return column.rolling(window=length, min_periods=length).min()

def is_max_over_window(column, length):
    return column == max_over_window(column, length)

def is_min_over_window(column, length):
    return column == min_over_window(column, length)
