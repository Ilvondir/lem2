import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import LEM2
from discretizer import Discretizer

pd.set_option("display.max_columns", 100)

train_data = pd.read_csv("train.csv").drop(["PassengerId", "Name", "Cabin"], axis=1)

test_data = pd.read_csv("test.csv")

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# To discretize: ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

discretizer = Discretizer()
discretizer.fit(train_data, ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], number_of_output_values=3, distance_from_extreme_values=.1, verbose=1)

train_data = discretizer.discretize(train_data)
test_data = discretizer.discretize(test_data)

print(train_data)

classifier = LEM2()
classifier.fit(train_data.drop("Transported", axis=1), train_data["Transported"], only_certain=False)
classifier.evaluate(train_data, train_data["Transported"])


test_data["Transported"] = classifier.predict(test_data)
test_data["Transported"] = test_data["Transported"].astype(bool)
test_data[["PassengerId", "Transported"]].to_csv("preds.csv", index=False)