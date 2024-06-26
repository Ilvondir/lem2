import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import *


# Teraz możesz zaimportować moduł

data = pd.read_csv('heart.csv')
labels = data['target']
data = data.drop('target', axis=1)

# print("Describe:")
# print(data.describe().transpose())

# data.hist()
# plt.show()

# scatter_matrix(data[['trestbps', 'chol','thalach', 'oldpeak']])
# plt.show()

# Data discretization
# We must discretize on attributes ['trestbps', 'chol','thalach', 'oldpeak']
# We will perform unsupervised discretization on this data. dividing the data into 5 interval values.

def discretization(x, column, number_of_groups):
    min_value = min(data[column])
    max_value = max(data[column])
    
    step = (max_value - min_value) / number_of_groups
    cuts = []
   
    
    for i in range(number_of_groups+1):
        cuts.append(min_value + step*i)
    
    for i in range(number_of_groups):
        if cuts[i] <= x < cuts[i+1]:
            return f"{cuts[i]}-{cuts[i+1]}"
        
    return f"{cuts[i]}-{cuts[i+1]}"

discretized_data = data.copy()

for column in ['age', 'trestbps', 'chol','thalach', 'oldpeak']:
    discretized_data[column] = discretized_data[column].apply(lambda x: discretization(x, column, 2))

    
# print(data)
# print(discretized_data)
# print(discretized_data['label'].value_counts())
    

discretized_data['label'] = labels
test_data = discretized_data.sample(n=200).reset_index(drop=True)
train_data = discretized_data.drop(test_data.index).reset_index(drop=True)

# print(train_data['label'].value_counts())

# # Classifier
lem2_classifier = LEM2()
lem2_classifier.fit(train_data.drop('label', axis=1), train_data['label'], only_certain=False)

# lem2_classifier.print_rules()

print("\n\nTrain data: ")
lem2_classifier.evaluate(train_data.drop('label', axis=1), train_data['label'])
# train_data['preds'] = lem2_classifier.predict(train_data.drop('label', axis=1), verbose=0)

print("\n\nTest data: ")
lem2_classifier.evaluate(test_data.drop('label', axis=1), test_data['label'])