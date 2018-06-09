
class TradingDecision:
    def __init__(self, description, action):
        self.description = description
        self.action = action

tradingDecisions = {
    "Buy": new TradingDecision("Buy", 1.0),
    "Sell": new TradingDecision("Sell", -1.0),
    "Neutral": new TradingDecision("Neutral", 0.0)
}
