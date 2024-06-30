import pandas as pd
import re
import string
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from discretizer import Discretizer
from lem2 import LEM2

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

data = pd.read_csv('Restaurant_Reviews.tsv', sep="\t")
data.columns = ['review', 'label']


def feature_engineering(review):
    # length of review
    length = len(review)
    
    # Number of words
    words_number = len(re.findall(r'\w+', review))
    
    # Mean word length
    mean_word_length = len(re.findall(r'\w', review)) / words_number
    
    # Percentage of punctuation marks
    punctuation_percentage = 0
    for c in review:
        if c in string.punctuation: punctuation_percentage += 1
        
    punctuation_percentage = punctuation_percentage / length * 100
       
    return length, words_number, mean_word_length, punctuation_percentage

data[['length', 'words_number', 'mean_word_length', 'punctuation_percentage']] = data['review'].apply(lambda review: pd.Series(feature_engineering(review)))
data['label'] = data.pop('label')


discretizer = Discretizer()

train_data = data.sample(n=750)
test_data = data.drop(train_data.index).reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

print(train_data)

print(f"Train shape: {train_data.shape}, test shape: {test_data.shape}")

discretizer = Discretizer()
discretizer.fit(train_data, ['length'], number_of_output_values=10)
discretizer.fit(train_data, ['words_number'], number_of_output_values=10)
discretizer.fit(train_data, ['mean_word_length'], number_of_output_values=10)
discretizer.fit(train_data, ['punctuation_percentage'], number_of_output_values=10, verbose=1)
# train_data = discretizer.discretize(train_data, decimal_places=2)
# test_data = discretizer.discretize(test_data, decimal_places=2)

print(test_data)


lem2 = LEM2()
lem2.fit(train_data.drop(["review", "label"], axis=1), train_data['label'], only_certain=False)
# lem2.print_rules()
lem2.evaluate(train_data.drop(["review", "label"], axis=1), train_data['label'])
lem2.evaluate(test_data.drop(["review", "label"], axis=1), test_data['label'])