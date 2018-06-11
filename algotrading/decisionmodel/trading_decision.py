
class TradingDecision:
    def __init__(self, description, action):
        self.description = description
        self.action = action

tradingDecisions = {
    "Buy": TradingDecision("Buy", 1.0),
    "Sell": TradingDecision("Sell", -1.0),
    "Neutral": TradingDecision("Neutral", 0.0)
}
