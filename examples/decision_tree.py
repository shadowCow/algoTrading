from sklearn import tree
from sklearn import metrics  
import numpy as np

import algotrading.data.feature as feature_engineering
import algotrading.data.features as features
import algotrading.data.markets as markets
import algotrading.tactics.tactics as tactics
import algotrading.evaluation_pipeline as evaluation_pipeline

data_directory = "futures_price_data"
markets_data = list(map((lambda m: evaluation_pipeline.load_data(data_directory, m)), markets.markets))

my_features = [
    # volatility
    features.is_last_above_average(features.oc_range, 5),

    # price change
    # - continuous variables - normalized by volatility
    features.last_to_average_ratio(features.oc_range, 5),

    # patterns
    # - binary variables indicating presence of patterns
    features.oc_is_up,
    features.is_max_feature(features.raw_data_as_feature("h"), 5),
    features.is_max_feature(features.raw_data_as_feature("c"), 5),
    features.is_min_feature(features.raw_data_as_feature("l"), 5),
    features.is_min_feature(features.raw_data_as_feature("c"), 5),
]

markets_with_features = feature_engineering.apply_features_to_markets(
    my_features,
    markets_data
)

# just corn, as an example
corn_data = markets_with_features[3]['data']

# get X from the feature columns...
feature_names = list(map((lambda f: f.name), my_features))
X = corn_data.loc[corn_data.index.year == 1990][feature_names]
print(X)

# target variable
open_to_open = tactics.normalized_open_to_open(corn_data, features.average_oc_range(5))
y = open_to_open.loc[corn_data.index.year == 1990]
print(y)

# split into test and train data...

# fit the model
decisionTree = tree.DecisionTreeRegressor(min_samples_leaf=10)
decisionTree.fit(X,y)
print("===== model =====")
print(decisionTree)

# make predictions / score.
X_test = corn_data.loc[corn_data.index.year == 1991][feature_names]
y_pred = decisionTree.predict(X_test)
print("===== result =====")
print(y_pred)

y_test = open_to_open.loc[corn_data.index.year == 1991]

model_score = decisionTree.score(X_test, y_test)
print("===== score =====")
print(model_score)

print("===== metrics =====")
print("y_train mean:", y.mean())
print("y_test mean:", y_test.mean())
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# print the tree
n_nodes = decisionTree.tree_.node_count
children_left = decisionTree.tree_.children_left
children_right = decisionTree.tree_.children_right
feature = decisionTree.tree_.feature
threshold = decisionTree.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))