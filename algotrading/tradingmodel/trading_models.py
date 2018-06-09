from . import trading_model

def tactics_with_decision_model(tactics, decision_model):
    """
    This can be used with static tactics.
    By static tactics, we mean that the computation of
    entry and exit values is fixed, and no decisions
    need to be made to decide those points.

    e.g.
    'open to open' trading is a static tactic.
    There is no decision to be made each period about what prices
    to use for entry and exit.

    The decision model here just looks at features and
    decides trading action without considering the tactics.
    """
    def trade_simulator(df, feature_columns):
        out_df = df.assign(trading_action=decision_model(df, feature_columns))
        out_df = out_df.assign(outcome=tactics(df) * out_df.trading_action)
        return out_df

    return trading_model.TradingModel(trade_simulator)
