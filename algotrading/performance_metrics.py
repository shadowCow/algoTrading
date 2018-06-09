
class ResultsSingleMarket:
    def __init__(self, total_net_gain, expectation, max_drawdown, num_trades):
        self.total_net_gain = total_net_gain
        self.expectation = expectation
        self.max_drawdown = max_drawdown
        self.num_trades = num_trades

    def __str__(self):
        return '\ntotal_net_gain: {},\nexpectation: {},\nmax_drawdown: {},\nnum_trades: {}'.format(
            self.total_net_gain,
            self.expectation,
            self.max_drawdown,
            self.num_trades
        )

    __repr__ = __str__

class ResultsAcrossMarkets:
    def __init__(self, agg_net_gain, agg_expectation, agg_max_drawdown, agg_num_trades, pct_markets_profitable):
        self.agg_net_gain = agg_net_gain
        self.agg_expectation = agg_expectation
        self.agg_max_drawdown = agg_max_drawdown
        self.agg_num_trades = agg_num_trades
        self.pct_markets_profitable = pct_markets_profitable

    def __str__(self):
        return '\nagg_net_gain: {},\nagg_expectation: {},\nagg_max_drawdown: {},\nagg_num_trades: {},\npct_markets_profitable: {}'.format(
            self.agg_net_gain,
            self.agg_expectation,
            self.agg_max_drawdown,
            self.agg_num_trades,
            self.pct_markets_profitable
        )

    __repr__ = __str__

class ResultsAcrossParameters:
    def __init__(self, agg_net_gain, agg_expectation, agg_max_drawdown, agg_num_trades, pct_markets_profitable):
        self.agg_net_gain = agg_net_gain
        self.agg_expectation = agg_expectation
        self.agg_max_drawdown = agg_max_drawdown
        self.agg_num_trades = agg_num_trades
        self.pct_markets_profitable = pct_markets_profitable

class AggregatedMetric:
    def __init__(self, avg, stddev, max, min):
        self.avg = avg
        self.stddev = stddev
        self.max = max
        self.min = min

    def __str__(self):
        return '\navg: {},\nstddev: {},\nmax: {},\nmin: {}'.format(
            self.avg,
            self.stddev,
            self.max,
            self.min
        )

    __repr__ = __str__
