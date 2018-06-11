import unittest
import pandas as pd
import numpy as np

from .context import algotrading
from algotrading.decisionmodel.random_decision import random_decision_model
from algotrading.decisionmodel.decision_model import make_decisions
from algotrading.decisionmodel.decision_model import decision_model_schemas

decision_column = decision_model_schemas["v1"].trading_decision
dates = pd.date_range(20010101, periods=5)
feature_columns = ["a"]


class RandomDecisionTests(unittest.TestCase):

    def test_long_only(self):
        test_df = RandomDecisionTests.get_test_data_frame()

        test_df = make_decisions(
            test_df,
            feature_columns,
            random_decision_model["long_only"]
        )

        for date in range(test_df.size):
            self.assertTrue(test_df.iat[date, df.columns.get_loc(decision_column)] == 1 or
                            test_df.at[date, df.columns.get_loc(decision_column)] == 0)

    def test_short_only(self):
        test_df = RandomDecisionTests.get_test_data_frame()

        test_df = make_decisions(
            test_df,
            feature_columns,
            random_decision_model["short_only"]
        )

        for date in range(test_df.size):
            self.assertTrue(test_df.at[date,df.columns.get_loc(decision_column)] == -1 or
                            test_df.at[date,df.columns.get_loc(decision_column)] == 0)

    def test_long_or_short(self):
        test_df = RandomDecisionTests.get_test_data_frame()

        test_df = make_decisions(
            test_df,
            feature_columns,
            random_decision_model["long_or_short"]
        )

        for date in range(test_df.size):
            self.assertTrue(test_df.at[date, df.columns.get_loc(decision_column)] == -1 or
                            test_df.at[date, df.columns.get_loc(decision_column)] == 0 or
                            test_df.at[date, df.columns.get_loc(decision_column)] == 1)

    @staticmethod
    def get_test_data_frame():
        return pd.DataFrame(
            np.random.(5,1),
            index=dates,
            columns=['a']
        )
