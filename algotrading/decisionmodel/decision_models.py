
def all_binary_true(long_or_short):
    def decider(df, feature_columns):
        # all feature_columns should be 1.0
        result = df[feature_columns].sum(axis=1)
        return result.map(lambda x: long_or_short if x == len(feature_columns) else 0)

    return decider

all_true_decision_model = {
    "long_only": all_binary_true(1),
    "short_only": all_binary_true(-1)
}
