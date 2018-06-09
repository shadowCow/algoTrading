import unittest

from .context import algotrading
from algotrading import evaluation_pipeline
import algotrading.data.markets as markets


class TestAlgoTrading(unittest.TestCase):

    def test_get_thing(self):
        self.assertEqual("hello", evaluation_pipeline.get_thing())


if __name__ == '__main__':
    unittest.main()
