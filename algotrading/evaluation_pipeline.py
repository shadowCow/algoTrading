import pandas
import numpy
import os.path as path
import sys
import algotrading.data.markets as markets
import algotrading.data.feature as feature_engineering
import algotrading.data.features as features
import algotrading.tactics.tactics as tactics
import algotrading.tradingmodel.trading_models as trading_models
import algotrading.tradingmodel.trading_model as trading_model
import algotrading.decisionmodel.decision_models as decision_models


def run_pipeline(markets_data, my_features, my_trading_model):
    markets_with_features = feature_engineering.apply_features_to_markets(
        my_features,
        markets_data
    )

    results = trading_model.evaluate_trading_model_multiple_markets(
        markets_with_features,
        [f.name for f in my_features],
        my_trading_model
    )

    return results


def load_data(data_directory, market):
    filename = market.symbol + "_Daily.txt"
    file_path = path.join(data_directory, filename)
    df = pandas.read_csv(
        file_path,
        delim_whitespace=True,
        header=1,
        names=['date', 'time', 'o', 'h', 'l', 'c', 'atr', 'volume'],
        usecols=['date', 'o', 'h', 'l', 'c'],
        dtype={
            'o': numpy.float64,
            'h': numpy.float64,
            'l': numpy.float64,
            'c': numpy.float64
        },
        index_col=0,
        parse_dates=True
    )
    return {
        "market": market,
        "data": df,
    }


def main():
    data_directory = sys.argv[1]
    markets_data = list(map((lambda m: load_data(data_directory, m)), markets.markets))

    my_features = [features.oc_is_down]
    my_trading_model = trading_models.tactics_with_decision_model(
        tactics.hold_open_to_close,
        decision_models.all_true_decision_model["long_only"]
    )
    results = run_pipeline(markets_data, my_features, my_trading_model)
    print(results)
    # maybe store results somewhere next...


if __name__ == '__main__':
    main()

