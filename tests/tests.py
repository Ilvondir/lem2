import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import *

data = pd.DataFrame({
    "a": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "b": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    "c": [1, 2, 2, 1, 1, 1, 1, 0, 0, 2],
    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}, dtype=int)

# data = pd.read_csv("./patient_statistics_discretized.csv").rename(columns={"Has_Disease": "label"})

# data = pd.DataFrame({
#     "a": [0, 0, 1, 1],
#     "b": [0, 1, 0, 1],
#     "label": [0, 1, 1, 0]
# }, dtype=int)

# test_data = pd.DataFrame({
#     "a": [2, 1],
#     "b": [0, 1],
#     "label": [1, 0],
# }, dtype=int)

lem2 = LEM2()
lem2.fit(data.drop('label', axis=1), data['label'], only_certain=False)
# lem2.evaluate(data.drop('label', axis=1), data['label'])
data['preds'] = lem2.predict(data, verbose=2)

print(data)


# lem2.print_rules()

# lem2.evaluate(data.drop('label', axis=1), data['label'])

# data['preds'] = lem2.predict(data, verbose=0)

# print(data)

# lem2.predict_object_class(data.iloc[43].to_dict(), lem2.rules, verbose=2)

