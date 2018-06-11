class DecisionModelSchema:
    def __init__(self, trading_decision):
        self.trading_decision = trading_decision


decision_model_schemas = {
    "v1": DecisionModelSchema("trading_decision")
}


def make_decisions(df, feature_columns, decision_function):
    decision_column = decision_function(df, feature_columns)
    df[decision_model_schemas["v1"].trading_decision] = decision_column
    return df
