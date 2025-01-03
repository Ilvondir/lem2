import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import LEM2
from discretizer import Discretizer


data = pd.read_csv('heart.csv')
labels = data['target']
data = data.drop('target', axis=1)

print(f"Data shape: {data.shape}")

# print("Describe:")
# print(data.describe().transpose())

# data.hist()
# plt.show()

# scatter_matrix(data[['trestbps', 'chol','thalach', 'oldpeak']])
# plt.show()

# Data discretization
# We must discretize on attributes ['trestbps', 'chol','thalach', 'oldpeak']
# We will perform unsupervised discretization on this data. dividing the data into 5 interval values.

data['label'] = labels
test_data = data.sample(n=250, random_state=55)
train_data = data.drop(test_data.index).reset_index(drop=True)
test_data = test_data.reset_index(drop=True)


discretizer = Discretizer()
discretizer.fit(train_data, ['trestbps', 'chol','thalach', 'oldpeak', 'age'], number_of_output_values=7, distance_from_extreme_values=0.15, verbose=1)

test_data = discretizer.discretize(test_data)
train_data = discretizer.discretize(train_data)

print(train_data)
print(test_data)


# Classifier
lem2_classifier = LEM2()

lem2_classifier.fit(train_data.drop('label', axis=1), train_data['label'], only_certain=False)

print("\n\nTrain data: ")
lem2_classifier.evaluate(train_data.drop('label', axis=1), train_data['label'])

print("\n\nTest data: ")
lem2_classifier.evaluate(test_data.drop('label', axis=1), test_data['label'])