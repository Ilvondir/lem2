import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from discretizer import Discretizer
from lem2 import *
import numpy as np

test_data = pd.read_csv("./test.csv").drop(["Name", "Ticket"], axis=1)
train_data = pd.read_csv("./train.csv").drop(['PassengerId', "Name", "Ticket"], axis=1)

print(f"train_data.shape: {train_data.shape}")
print(f"test_data.shape: {test_data.shape}")

disc = Discretizer()
disc.fit(train_data, ['Age'], number_of_output_values=9, distance_from_extreme_values=.05)
disc.fit(train_data, ['Fare'], number_of_output_values=9, distance_from_extreme_values=.05, verbose=1)

train_data = disc.discretize(train_data)
test_data = disc.discretize(test_data)

print(train_data['Survived'].value_counts())

# print(train_data)

# lem2 = LEM2()
# lem2.fit(train_data.drop(["Survived"], axis=1), train_data["Survived"], only_certain=True, verbose=1)
# lem2.evaluate(train_data.drop(["Survived"], axis=1), train_data["Survived"])

# test_data["Survived"] = lem2.predict(test_data)
# test_data["Survived"] = test_data["Survived"].astype(int)
# test_data[["PassengerId", "Survived"]].to_csv("preds.csv", index=False)