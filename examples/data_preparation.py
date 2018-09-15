import pandas as pd
import numpy as np

from algotrading.data.price_data_schema import price_data_schema as schema

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

preparation_pipeline = Pipeline([
    
])