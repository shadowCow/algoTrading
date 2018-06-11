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
        print(df_one)
        print(df_two)
        markets = [
            {"market": {"symbol": "A"},
             "data": df_one},
            {"market": {"symbol": "B"},
             "data": df_two}
        ]

        my_features = [features.oc_change]
        my_trading_model = algotrading.tradingmodel.trading_models.tactics_with_decision_model(
            tactics.hold_open_to_close,
            decision_models.all_true_decision_model["long_only"]
        )

        results = evaluation_pipeline.run_pipeline(
            markets,
            my_features,
            my_trading_model
        )
        print(results)


if __name__ == '__main__':
    unittest.main()
