from algotrading.data.feature import Feature, VariableTypes

oc_change = Feature(
    'oc_change',
    VariableTypes.continuous,
    lambda df: df.c - df.o
)

oc_is_up = Feature(
    'oc_is_up',
    VariableTypes.binary,
    lambda df: df.c > df.o
)

oc_is_down = Feature(
    'oc_is_down',
    VariableTypes.binary,
    lambda df: df.c < df.o
)
