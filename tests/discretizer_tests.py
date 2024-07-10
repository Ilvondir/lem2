import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from discretizer import *
import pandas as pd

data = pd.DataFrame({
    'a': [1.2, 2.2, 4.2, 1, -1, 3.4],
    'b': [6.2, 7.7, 3.3, 2.3, .6, .9],
    'c': [6.6, 5.4, 2.1, 0.5, -1.4, 3.4]
}, dtype=float) 

data2 = pd.DataFrame({
    'a': [1.2, 2.2, 4.2, 1, -1, 3.4],
    'b': [6.2, 7.7, 3.3, 2.3, .6, .9],
    'c': [6.6, 5.4, 2.1, 0.5, -1.4, 3.4]
}, dtype=float) 

disc = Discretizer()

print(data)

disc.fit(data, data.columns, 3, distance_from_extreme_values=0.4, verbose=1)

print(data2)
print(disc.discretize(data2))