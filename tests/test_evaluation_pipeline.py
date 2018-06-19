import unittest
import pandas
import numpy

from .context import algotrading
from .context import get_test_data_frame_one
from .context import get_test_data_frame_two
import algotrading.data.features as features
import algotrading.decisionmodel.decision_models as decision_models
import algotrading.tactics.tactics as tactics
import algotrading.tradingmodel.trading_models as trading_models
import algotrading.evaluation_pipeline as evaluation_pipeline

class EvaluationPipelineTests(unittest.TestCase):

    def test_evaluation_pipeline(self):
        df_one = get_test_data_frame_one()
        df_two = get_test_data_frame_two()
        
        markets = [
            {"market": {"symbol": "A"},
             "data": df_one},
            {"market": {"symbol": "B"},
             "data": df_two}
        ]

        my_features = [features.oc_is_up]
        my_trading_model = algotrading.tradingmodel.trading_models.tactics_with_decision_model(
            tactics.hold_open_to_close,
            decision_models.all_true_decision_model["long_only"]
        )

        results = evaluation_pipeline.run_pipeline(
            markets,
            my_features,
            my_trading_model
        )

        self.assertAlmostEqual(results.agg_net_gain.avg, 0.175)
        self.assertAlmostEqual(results.agg_net_gain.stddev, 1.591, places=3)
        self.assertAlmostEqual(results.agg_net_gain.max, 1.3)
        self.assertAlmostEqual(results.agg_net_gain.min, -0.95)

        self.assertAlmostEqual(results.agg_expectation.avg, -0.258, places=3)
        self.assertAlmostEqual(results.agg_expectation.stddev, 0.978, places=3)
        self.assertAlmostEqual(results.agg_expectation.max, 0.433, places=3)
        self.assertAlmostEqual(results.agg_expectation.min, -0.95)

        self.assertAlmostEqual(results.agg_max_drawdown.avg, 0.975)
        self.assertAlmostEqual(results.agg_max_drawdown.stddev, 0.035, places=3)
        self.assertAlmostEqual(results.agg_max_drawdown.max, 1.0)
        self.assertAlmostEqual(results.agg_max_drawdown.min, 0.95)

        self.assertAlmostEqual(results.agg_num_trades.avg, 2.0)
        self.assertAlmostEqual(results.agg_num_trades.stddev, 1.414, places=3)
        self.assertAlmostEqual(results.agg_num_trades.max, 3)
        self.assertAlmostEqual(results.agg_num_trades.min, 1)

        self.assertAlmostEqual(results.pct_markets_profitable, 0.5)


if __name__ == '__main__':
    unittest.main()
