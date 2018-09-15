import pandas as pd

class MarketDirection:
    def __init__(self, up, down, flat):
        self.up = up
        self.down = down
        self.flat = flat


market_direction = MarketDirection("up", "down", "flat")

def get_feature_direction(feature):
    def direction(value):
        if (value > 0.0):
            return market_direction.up
        elif (value < 0.0):
            return market_direction.down
        else:
            return market_direction.flat
    
    return feature.apply(direction)
    