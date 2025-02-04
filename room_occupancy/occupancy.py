import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import LEM2
from discretizer import Discretizer

data = pd.read_csv('file.csv')
labels = data['Occupancy']
data = data.drop('Occupancy', axis=1)

print(f"Data shape: {data.shape}")

data['label'] = labels
test_data = data.sample(n=400, random_state=42)
train_data = data.drop(test_data.index).reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

discretizer = Discretizer()
discretizer.fit(train_data, ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'], number_of_output_values=50, distance_from_extreme_values=.2, verbose=1)
test_data = discretizer.discretize(test_data)
train_data = discretizer.discretize(train_data)

print(train_data)
print(test_data)

lem2_classifier = LEM2()

lem2_classifier.fit(train_data.drop('label', axis=1), train_data['label'], only_certain=False)

print("\n\nTrain data: ")
lem2_classifier.evaluate(train_data.drop('label', axis=1), train_data['label'])

print("\n\nTest data: ")
lem2_classifier.evaluate(test_data.drop('label', axis=1), test_data['label'])