# LEM2

The implementation of the LEM2 algorithm (Learning from Examples Module, version 2), a greedy machine learning algorithm used in classification problems. It is based on the theory of rough sets and works by generating a minimal covering of decision rules for each label in the training data.

The implementation allows the algorithm to run in various debugging modes using the verbose parameter. Additionally, there is an option to choose whether the algorithm should generate only certain rules or also uncertain ones, using the only_certain parameter.

Due to the fact that the LEM2 algorithm works only with discretized data, a simple discretizer has also been implemented. Its operation is based on dividing the range of a given attribute into a selected number of sets, which represent discrete value.

The proposed implementation was developed in Python and tested on several different binary classification problems.

## Used Tools

- Python 3.11.2
- Pandas 2.2.3
- Matplotlib 3.10.0
- Numpy 2.2.0

## Requirements

For running the application you need:

- [Python](https://www.python.org/downloads/)

## How to run

1. Execute command `git clone https://github.com/Ilvondir/lem2`.
2. Install required packages by `pip install -r requirements.txt`.
3. Check the implementation in  `lem2.py`, its tests and documentations in `html` files.
