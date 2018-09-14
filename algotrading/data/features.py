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

oc_range = Feature(
    'oc_range',
    VariableTypes.continuous,
    lambda df: (df.c - df.o).abs()
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

# multi-bar stuff

# volatility
oc_range_is_up = Feature(
    'oc_range_is_up',
    VariableTypes.binary,
    lambda df: oc_range.transform(df)
)

def average_oc_range(length):
    return moving_average_feature(oc_range, length)

def average_hl_range(length):
    return moving_average_feature(hl_range, length)



# window features
def change_over_time_feature(feature, length):
    """
    Compute the change in a given feature between the current value,
    and a value (length-1) periods previous.

    e.g. Given a 2 period feature f1 with values: [1.0, 3.0],
    change_over_time_feature(f1, 2) would yield 2.0
    """
    transformed = feature.transform(df)
    return Feature(
        'cot_{}_{}'.format(feature.name, length),
        VariableTypes.continuous,
        lambda df: transformed - transformed.shift(length-1)
    )

def range_over_time_feature(feature, length):
    """
    Compute the range in a given feature between the current value,
    and a value (length-1) periods previous.
    Since it is a range - all results will be >= 0.

    e.g. Given a 2 period feature f1 with values: [4.0, 3.0],
    range_over_time_feature(f1, 2) would yield 1.0
    """
    transformed = feature.transform(df)
    return Feature(
        'rot_{}_{}'.format(feature.name, length),
        VariableTypes.continuous,
        lambda df: (transformed - transformed.shift(length-1)).abs()
    )

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

def is_last_above_average(feature, length):
    def transform(df):
        new_col = feature.transform(df).rename(feature.name)
        with_feature = pd.concat([df, new_col], axis=1)
        return with_feature[feature.name] > moving_average(with_feature[feature.name], length).shift(1)

    return Feature(
        'is_last_above_average_{}_{}'.format(length, feature.name),
        VariableTypes.binary,
        lambda df: transform(df)
    )

def is_last_below_average(feature, length):
    def transform(df):
        new_col = feature.transform(df).rename(feature.name)
        with_feature = pd.concat([df, new_col], axis=1)
        return with_feature[feature.name] < moving_average(with_feature[feature.name], length).shift(1)

    return Feature(
        'is_last_below_average_{}_{}'.format(length, feature.name),
        VariableTypes.binary,
        lambda df: transform(df)
    )

def last_to_average_ratio(feature, length):
    def transform(df):
        new_col = feature.transform(df).rename(feature.name)
        with_feature = pd.concat([df, new_col], axis=1)
        return with_feature[feature.name] / moving_average(with_feature[feature.name], length).shift(1)

    return Feature(
        'last_to_average_ratio_{}_{}'.format(length, feature.name),
        VariableTypes.continuous,
        lambda df: transform(df)
    )
    
# Helpers

def get_streak_counter_for_feature(df, feature):
    new_col = feature.transform(df).rename(feature.name)
    new_df = pd.concat([df, new_col], axis=1)
    return streak_counter(
        new_df,
        feature.name
    )

def streak_counter_feature(feature):
    return Feature(
        'streak_{}'.format(feature.name),
        VariableTypes.discrete,
        lambda df: get_streak_counter_for_feature(df, feature)
    )

def is_true_streak_feature(feature):
    def compute(df, feature):
        streak_counter = get_streak_counter_for_feature(df, feature)
        streak_counter[streak_counter > 0] = True
        streak_counter[streak_counter <= 0] = False
        return streak_counter

    return Feature(
        'is_true_streak_{}'.format(feature.name),
        VariableTypes.binary,
        lambda df: compute(df, feature)
    )

def is_false_streak_feature(feature):
    def compute(df, feature):
        streak_counter = get_streak_counter_for_feature(df, feature)
        streak_counter[streak_counter >= 0] = False
        streak_counter[streak_counter < 0] = True
        return streak_counter

    return Feature(
        'is_false_streak_{}'.format(feature.name),
        VariableTypes.binary,
        lambda df: compute(df, feature)
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
