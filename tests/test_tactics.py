import unittest
import pandas as pd
import numpy as np

from .context import algotrading
from .context import dates
from .context import assert_elements_equal
from .context import get_test_data_frame_one

import algotrading.tactics.tactics as tactics

class TacticsTests(unittest.TestCase):

    def test_hold_open_to_close(self):
        df = get_test_data_frame_one()

        open_to_close = tactics.hold_open_to_close(df)
        expected = pd.Series([1.0,1.3,-1.0,-1.2,float('nan')], index=dates)
        assert_elements_equal(self, expected, open_to_close)

    def test_hold_open_to_open(self):
        df = get_test_data_frame_one()

        open_to_open = tactics.hold_open_to_open(df)
        expected = pd.Series([1.1,0.0,-1.4,float('nan'),float('nan')], index=dates)
        assert_elements_equal(self, expected, open_to_open)
