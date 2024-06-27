import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from discretizer import Discretizer
from lem2 import *

data = pd.read_csv("./titanic.csv")

complete_data = data.dropna().reset_index(drop=True)
semantic_data = complete_data.drop(['name', 'ticketno'], axis=1)
    
# To discretize: ['age', 'fare']
    
    
test_data = semantic_data.sample(n=200).reset_index(drop=True)
train_data = semantic_data.drop(test_data.index).reset_index(drop=True)

discretizer = Discretizer()
discretizer.fit(train_data, ['age', 'fare'], 10, verbose=1)

test_data = discretizer.discretize(test_data, ['age', 'fare'])
train_data = discretizer.discretize(train_data, ['age', 'fare'])


lem2_classifier = LEM2()
lem2_classifier.fit(train_data.drop('survived', axis=1), labels=train_data['survived'], only_certain=False)

print("train:")
lem2_classifier.evaluate(train_data.drop('survived', axis=1), labels=train_data['survived'])

print("test:")
lem2_classifier.evaluate(test_data, test_data['survived'])

print(test_data)

lem2_classifier.print_rules()