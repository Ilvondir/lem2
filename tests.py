from lem2 import *
import pandas as pd

# data = pd.DataFrame({
#     "a": [0, 0, 0, 0, 0, 0, 1, 1, 0],
#     "b": [0, 1, 0, 0, 0, 1, 0, 0, 1],
#     "c": [1, 2, 2, 1, 1, 1, 1, 0, 0],
#     "label": [0, 0, 0, 0, 0, 1, 1, 1, 1]
# }, dtype=int)

data = pd.read_csv("./patient_statistics_discretized.csv").rename(columns={"Has_Disease": "label"})

# data = pd.DataFrame({
#     "a": [0, 0, 1, 1, 1],
#     "b": [0, 1, 0, 1, 1],
#     "label": [0, 1, 1, 0, 1]
# }, dtype=int)

# test_data = pd.DataFrame({
#     "a": [2, 1],
#     "b": [0, 1],
#     "label": [1, 0],
# }, dtype=int)

lem2 = LEM2()
lem2.fit(data.drop('label', axis=1), data['label'], only_certain=True, verbose=1)
# lem2.evaluate(data.drop('label', axis=1), data['label'])
lem2.predict(data)


# lem2.print_rules()

# lem2.evaluate(data.drop('label', axis=1), data['label'])

# data['preds'] = lem2.predict(data, verbose=0)

# print(data)

# lem2.predict_object_class(data.iloc[43].to_dict(), lem2.rules, verbose=2)