import random


def random_decision_long(df, feature_columns):
    return df.apply(lambda row: random.randint(0, 1), axis=1)


def random_decision_short(df, feature_columns):
    return df.apply(lambda row: random.randint(-1, 0), axis=1)


def random_decision_long_or_short(df, feature_columns):
    return df.apply(lambda row: random.randint(-1, 1), axis=1)


random_decision_model = {
    "long_only": random_decision_long,
    "short_only": random_decision_short,
    "long_or_short": random_decision_long_or_short
}
