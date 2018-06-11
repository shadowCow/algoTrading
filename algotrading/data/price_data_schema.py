class PriceDataSchema:
    def __init__(self, date, open, high, low, close):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close


priceDataSchemas = {
    "v1": PriceDataSchema("date", "o", "h", "l", "c")
}
