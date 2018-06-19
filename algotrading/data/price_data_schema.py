class PriceDataSchema:
    def __init__(self, date, open, high, low, close, true_range):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.true_range = true_range


priceDataSchemas = {
    "v1": PriceDataSchema("date", "o", "h", "l", "c", "tr")
}
