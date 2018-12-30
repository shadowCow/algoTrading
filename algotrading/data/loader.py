import pandas as pd
import numpy as np
import os.path as path


def load_daily_data(data_directory, market):
    filename = market.symbol + "_Daily.txt"
    file_path = path.join(data_directory, filename)
    df = load_data_file(file_path)

    return {
        "market": market,
        "data": df,
    }


def load_data_file(file_path):
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=1,
        names=['date', 'time', 'o', 'h', 'l', 'c', 'tr', 'volume'],
        usecols=['date', 'o', 'h', 'l', 'c', 'tr'],
        dtype={
            'o': np.float64,
            'h': np.float64,
            'l': np.float64,
            'c': np.float64,
            'tr': np.float64
        },
        index_col=0,
        parse_dates=True
    )

    return df


def load_all_markets(data_directory, markets):
    return list(map((lambda m: load_daily_data(data_directory, m)), markets))

