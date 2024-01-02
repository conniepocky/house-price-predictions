from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 

train_data = pd.read_csv("data/train.csv")

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr"]

y = train_data.SalePrice

X = train_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)
    return(mae)

scores = {}

for max_leaf_nodes in range(5, 1000, 1):
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    scores[max_leaf_nodes] = mae

optimum_leaf_nodes = min(scores, key=scores.get)

print(optimum_leaf_nodes)

model = DecisionTreeRegressor(max_leaf_nodes=optimum_leaf_nodes, random_state=0)

model.fit(X,y)

val_predictions = model.predict(val_X)

print("mae:", mean_absolute_error(val_y, val_predictions))

#Predicting on test data

test_data = pd.read_csv("data/test.csv")

test_X = test_data[features]

test_predictions = model.predict(test_X)

output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_predictions})
output.to_csv("predictions.csv", index=False)