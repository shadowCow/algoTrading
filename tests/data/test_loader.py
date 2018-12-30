import unittest
import os.path as path

from tests.context import algotrading

from algotrading.data.loader import load_data_file
from algotrading.data.loader import load_daily_data
from algotrading.data.markets import markets

class TestLoader(unittest.TestCase):

    def test_load_data_file(self):
        file_path = path.join(
            "futures_price_data",
            "C_Daily.txt"
        )

        df = load_data_file(file_path)

        self.assertEqual(7486, df.shape[0])
        self.assertEqual(5, df.shape[1])


    def test_load_daily_data(self):
        data_directory = "futures_price_data"
        market = markets[0]

        market_with_data = load_daily_data(data_directory, market)

        self.assertEqual(market_with_data["market"].symbol, "AD")
        self.assertEqual(market_with_data["market"].long_name, "Australian Dollar")
        self.assertEqual(market_with_data["data"].shape[0], 6218)
        self.assertEqual(market_with_data["data"].shape[1], 5)

        