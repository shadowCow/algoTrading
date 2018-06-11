"""
Tactics represent details of trade execution.

Trade execution happens in the future relative to decisions.

Any tactics should be lagged appropriately,
otherwise the trading model is 'cheating' by looking
into the future before it happens.
"""


def hold_open_to_close(df):
    return df.c.shift(-1) - df.o.shift(-1)
