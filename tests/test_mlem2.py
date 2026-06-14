import unittest

import pandas as pd

from mlem2 import MLEM2


class MLEM2Tests(unittest.TestCase):
    def test_example_from_lecture(self):
        data = pd.DataFrame(
            {
                "a1": [0.9, 2.7, 2.7, 1.2, 3.3, 4.3, 1.2],
                "a2": [6.2, 6.2, 2.1, 4.5, 6.8, 4.5, 6.2],
            }
        )
        labels = pd.Series([3, 2, 1, 2, 2, 1, 3])

        classifier = MLEM2()
        classifier.fit(data, labels, only_certain=True, verbose=0)

        self.assertEqual(
            classifier.cut_points_,
            {
                "a1": [1.05, 1.95, 3.0, 3.8],
                "a2": [3.3, 5.35, 6.5],
            },
        )
        self.assertEqual(
            classifier.rules[0],
            [
                ["a1", "<", 1.95, 3],
                ["a2", ">=", 5.35, 3],
            ],
        )
        self.assertEqual(classifier.predict(data, verbose=0), labels.tolist())

    def test_symbolic_and_numeric_attributes(self):
        data = pd.DataFrame(
            {
                "age": [20, 22, 40, 42],
                "color": ["red", "red", "blue", "blue"],
            }
        )
        labels = pd.Series(["young", "young", "old", "old"])

        classifier = MLEM2()
        classifier.fit(data, labels, verbose=0)

        self.assertEqual(classifier.predict(data, verbose=0), labels.tolist())
        self.assertEqual(
            classifier.evaluate(data, labels, verbose=0),
            ("accuracy", 1.0),
        )

    def test_lower_and_upper_approximations(self):
        data = pd.DataFrame({"a": [0, 0, 1, 2]})
        labels = pd.Series(["yes", "no", "yes", "no"])

        certain = MLEM2()
        certain.fit(data, labels, only_certain=True, verbose=0)

        possible = MLEM2()
        possible.fit(data, labels, only_certain=False, verbose=0)

        certain_yes_rules = [
            rule for rule in certain.rules if rule[0][-1] == "yes"
        ]
        possible_yes_rules = [
            rule for rule in possible.rules if rule[0][-1] == "yes"
        ]

        self.assertTrue(certain_yes_rules)
        self.assertTrue(possible_yes_rules)
        self.assertNotEqual(certain_yes_rules, possible_yes_rules)


if __name__ == "__main__":
    unittest.main()
