from sklearn.pipeline import Pipeline

from algotrading.data.loader import load_daily_data
from algotrading.data.markets import markets


market_with_data = load_daily_data("futures_price_data", markets[3])

derive_feature_steps = [
    
]

data_frame_pipeline = Pipeline(derive_feature_steps)
with_features = data_frame_pipeline.fit_transform(
    market_with_data["data"].copy()
)