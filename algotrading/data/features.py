from . import feature

oc_change = feature.Feature(
    'oc_change',
    feature.VariableTypes.continuous,
    lambda df: df.c - df.o
)

oc_is_up = feature.Feature(
    'oc_is_up',
    feature.VariableTypes.binary,
    lambda df: df.c > df.o
)

oc_is_down = feature.Feature(
    'oc_is_down',
    feature.VariableTypes.binary,
    lambda df: df.c < df.o
)
