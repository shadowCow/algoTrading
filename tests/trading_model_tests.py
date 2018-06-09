import unittest
import pandas
import numpy

from .context import algotrading
from .context import get_test_data_frame_one
from .context import get_test_data_frame_two
from .context import dummy_trade_simulator

import algotrading.tradingmodel.trading_model as trading_model

class TradingModelTests(unittest.TestCase):

    def test_evaluation_single_market(self):
        df = get_test_data_frame_one()

        my_model = trading_model.TradingModel(dummy_trade_simulator)

        results = trading_model.evaluate_trading_model_on_market(df, [], my_model)

        self.assertEqual(results.num_trades, 3)
        self.assertAlmostEqual(results.total_net_gain, 0.9)
        self.assertAlmostEqual(results.expectation, 0.3)
        self.assertAlmostEqual(results.max_drawdown, 1.2)

    def test_evaluation_multiple_markets(self):
        df_one = get_test_data_frame_one()
        df_two = get_test_data_frame_two()

        markets = [
            {"market": {"symbol": "A"},
             "data": df_one},
            {"market": {"symbol": "B"},
             "data": df_two}
        ]

        my_model = trading_model.TradingModel(dummy_trade_simulator)

        results = trading_model.evaluate_trading_model_multiple_markets(
            markets,
            [],
            my_model
        )

        self.assertAlmostEqual(results.agg_net_gain.avg, -0.725)
        self.assertAlmostEqual(results.agg_net_gain.stddev, 2.298, places=3)
        self.assertAlmostEqual(results.agg_net_gain.max, 0.9)
        self.assertAlmostEqual(results.agg_net_gain.min, -2.35)

        self.assertAlmostEqual(results.agg_expectation.avg, -0.2416, places=3)
        self.assertAlmostEqual(results.agg_expectation.stddev, 0.766, places=3)
        self.assertAlmostEqual(results.agg_expectation.max, 0.3)
        self.assertAlmostEqual(results.agg_expectation.min, -0.783, places=3)

        self.assertAlmostEqual(results.agg_max_drawdown.avg, 1.775, places=3)
        self.assertAlmostEqual(results.agg_max_drawdown.stddev, 0.813, places=3)
        self.assertAlmostEqual(results.agg_max_drawdown.max, 2.35)
        self.assertAlmostEqual(results.agg_max_drawdown.min, 1.2)

        self.assertAlmostEqual(results.agg_num_trades.avg, 3.0)
        self.assertAlmostEqual(results.agg_num_trades.stddev, 0.0)
        self.assertAlmostEqual(results.agg_num_trades.max, 3)
        self.assertAlmostEqual(results.agg_num_trades.min, 3)

        self.assertAlmostEqual(results.pct_markets_profitable, 0.5)
