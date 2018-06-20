from algotrading.data.feature import Feature, VariableTypes

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

oc_is_up_streak = Feature(
    'oc_is_up_streak',
    VariableTypes.discrete,
    lambda df: streak_counter(df, 'oc_is_up')
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
    return df.groupby((df[col_name] != df[col_name].shift(1)).cumsum()).cumcount()

def max_over_window(column, length):
    return column.rolling(window=length, min_periods=length).max()

def min_over_window(column, length):
    return column.rolling(window=length, min_periods=length).min()

def is_max_over_window(column, length):
    return column == max_over_window(column, length)

def is_min_over_window(column, length):
    return column == min_over_window(column, length)
