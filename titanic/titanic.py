import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lem2 import *

data = pd.read_csv("./titanic.csv")

complete_data = data.dropna().reset_index(drop=True)
semantic_data = complete_data.drop(['name', 'ticketno'], axis=1)

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

for attribute in ['age', 'fare']:
    semantic_data[attribute] = semantic_data[attribute].apply(lambda x: discretization(x, attribute, 5))
    
    
test_data = semantic_data.sample(n=200).reset_index(drop=True)
train_data = semantic_data.drop(test_data.index).reset_index(drop=True)


lem2_classifier = LEM2()
lem2_classifier.fit(train_data.drop('survived', axis=1), labels=train_data['survived'], only_certain=False)
# lem2_classifier.evaluate(train_data.drop('survived', axis=1), labels=train_data['survived'])

# lem2_classifier.evaluate(test_data, test_data['survived'])