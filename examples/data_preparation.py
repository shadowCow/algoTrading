import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from algotrading.data.price_data_schema import price_data_schema as schema
from algotrading.data.market_direction import market_direction as md
from algotrading.data.features import intra_bar_features as ibf
from algotrading.data.features import streak_features as stf
from algotrading.data.features import window_features as wf
from algotrading.data.features.feature_helpers import DataFrameSelector

dates = pd.date_range(20010101, periods=5)
df = pd.DataFrame(
        np.array([
            [1.1, 1.9, 1.1, 1.9],
            [2.3, 3.5, 2.1, 3.3],
            [3.4, 5.0, 3.0, 4.7],
            [3.4, 3.6, 2.2, 2.4],
            [2.0, 2.0, 0.7, 0.8]
        ]),
        index = dates,
        columns = [schema.open, schema.high, schema.low, schema.close]
    )

# derive features from the raw data and add them to the DataFrame
derive_features_steps = [
    ('oc_range', ibf.OpenCloseRange()),
    ('oc_direction', ibf.OpenCloseDirection()),
    ('direction_streaks', stf.StreakCounterFeature('oc_direction')),
    ('up_streaks', stf.StreakCounterByValueFeature('oc_direction', md.up)),
    ('average_oc_range', wf.Average("oc_range", 3)),
]

data_frame_pipeline = Pipeline(derive_features_steps)
with_features = data_frame_pipeline.fit_transform(df.copy())
print("==== DataFrame with features added ====")
print(with_features)

# transform the dataframe into the matrices we need for scikit learn algorithms
prepare_for_ml_steps = [
    # choose the features we want for our ml technique
    ('selector', DataFrameSelector(['oc_direction_up_streak', 'oc_range_average_3'])),
]

preparation_pipeline = Pipeline(derive_features_steps + prepare_for_ml_steps)
df_prepared = preparation_pipeline.fit_transform(df.copy())
print("==== Matrix for ML ====")
print(df_prepared)
