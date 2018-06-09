import pandas
import performance_metrics

class TradingModel:
    """A TradingModel converts market data into trades and trade outcomes.

    Args:
        trade_simulator (DataFrame => DataFrame): A function that takes the input
            data and returns that data frame with metrics attached to it.
            The function should apply the 'outcome' and 'trading_action' columns
            to the data frame.
    """
    def __init__(self, trade_simulator):
        self.trade_simulator = trade_simulator

    def simulate_trades(self, df, feature_columns):
        """Apply a trading model to the data for a single market.

        This will produce a data frame with all the performance metrics
        needed to evaluate the trading model.

        Args:
            df (DataFrame): The price data and any additional features.
            feature_columns (Array): An array of column names indicating which
                columns should be used by the trading model.

        Returns:
            DataFrame: A data frame containing the input data and results of simulation.

            The returned data frame will contain these additional columns:
            outcome - The gain/loss for applying to the trading model to the given row.
            trading_action - Which action was taken for the given row (buy, sell, neutral).
        """
        return self.trade_simulator(df, feature_columns)


def evaluate_trading_model_on_market(df, feature_columns, trading_model):
    outcome_df = trading_model.simulate_trades(df, feature_columns)
    print(outcome_df)
    outcome_column = outcome_df['outcome']

    # need to prepend 0.
    # if the net gain is always negative from the first bar,
    # max net gain will be negative, which is not true, because you start at 0.
    running_net_gain = pandas.Series([0]).append(outcome_column).cumsum()
    running_max_net_gain = running_net_gain.cummax()
    running_drawdown = running_max_net_gain - running_net_gain

    total_net_gain = running_net_gain.iat[-1]
    num_trades = outcome_df['trading_action'].sum()
    expectation = total_net_gain / num_trades if num_trades > 0 else 0
    max_drawdown = running_drawdown.max()

    return performance_metrics.ResultsSingleMarket(
        total_net_gain,
        expectation,
        max_drawdown,
        num_trades
    )

def evaluate_trading_model_multiple_markets(markets, feature_columns, trading_model):
    def get_results(market):
        results = evaluate_trading_model_on_market(
            market["data"],
            feature_columns,
            trading_model
        )
        return {
            "market": market["market"],
            "results": results
        }

    results_per_market = list(map(get_results, markets))

    return performance_metrics.ResultsAcrossMarkets(
        compute_aggregated_net_gain(results_per_market),
        compute_aggregated_expectation(results_per_market),
        compute_aggregated_max_drawdown(results_per_market),
        compute_aggregated_num_trades(results_per_market),
        compute_pct_markets_profitable(results_per_market)
    )


def compute_aggregated_net_gain(results_per_market):
    all_net_gains = pandas.Series(list(map((lambda x: x["results"].total_net_gain), results_per_market)))

    return compute_aggregated_metric(all_net_gains)

def compute_aggregated_expectation(results_per_market):
    all_expectations = pandas.Series(list(map((lambda x: x["results"].expectation), results_per_market)))

    return compute_aggregated_metric(all_expectations)

def compute_aggregated_max_drawdown(results_per_market):
    all_max_drawdowns = pandas.Series(list(map((lambda x: x["results"].max_drawdown), results_per_market)))

    return compute_aggregated_metric(all_max_drawdowns)

def compute_aggregated_num_trades(results_per_market):
    all_num_trades = pandas.Series(list(map((lambda x: x["results"].num_trades), results_per_market)))

    return compute_aggregated_metric(all_num_trades)

def compute_aggregated_metric(individual_values):
    return performance_metrics.AggregatedMetric(
        individual_values.mean(),
        individual_values.std(),
        individual_values.max(),
        individual_values.min()
    )

def compute_pct_markets_profitable(results_per_market):
    all_net_gains = pandas.Series(list(map((lambda x: x["results"].total_net_gain), results_per_market)))

    profitable_markets = list(filter(lambda m: m > 0, all_net_gains))

    return len(profitable_markets) / len(all_net_gains)
