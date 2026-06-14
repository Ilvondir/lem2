import sys
import time

import numpy as np
import pandas as pd


class LEM2:
    """NumPy-based implementation of the LEM2 rule classifier."""

    _MISSING = "__LEM2_MISSING__"

    def __init__(self) -> None:
        self.rules = []
        self.label_counts_ranking = []
        self.feature_names_ = []
        self._encoded_rules = []
        self._value_to_code = []
        self._code_to_value = []
        self._label_to_index = {}

    def _prepare_training_data(self, data, labels):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        labels = pd.Series(labels, copy=False).reset_index(drop=True)
        data = data.reset_index(drop=True).copy()
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")

        data = data.fillna(self._MISSING)
        combined = data.copy()
        combined["__lem2_label__"] = labels.to_numpy()
        combined = combined.drop_duplicates().reset_index(drop=True)

        labels = combined.pop("__lem2_label__")
        self.feature_names_ = list(combined.columns)
        self._value_to_code = []
        self._code_to_value = []
        encoded_columns = []

        for name in self.feature_names_:
            codes, values = pd.factorize(combined[name], sort=False)
            values = values.tolist()
            encoded_columns.append(codes.astype(np.int64, copy=False))
            self._value_to_code.append(
                {value: code for code, value in enumerate(values)}
            )
            self._code_to_value.append(values)

        if encoded_columns:
            encoded = np.column_stack(encoded_columns)
        else:
            encoded = np.empty((len(combined), 0), dtype=np.int64)

        return combined, encoded, labels.to_numpy()

    @staticmethod
    def _row_groups(encoded):
        if len(encoded) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if encoded.shape[1] == 0:
            return np.zeros(len(encoded), dtype=np.int64), np.array([len(encoded)])

        _, inverse, counts = np.unique(
            encoded, axis=0, return_inverse=True, return_counts=True
        )
        return inverse, counts

    @staticmethod
    def _rule_mask(encoded, rule):
        mask = np.ones(len(encoded), dtype=bool)
        for column, value_code in rule:
            mask &= encoded[:, column] == value_code
        return mask

    def _rule_is_sufficient(self, encoded, rule, approximation_mask):
        return not np.any(self._rule_mask(encoded, rule) & ~approximation_mask)

    def _minimalize_rule(self, encoded, rule, approximation_mask):
        minimized = list(rule)
        position = 0
        while position < len(minimized):
            candidate = minimized[:position] + minimized[position + 1 :]
            if candidate and self._rule_is_sufficient(
                encoded, candidate, approximation_mask
            ):
                minimized = candidate
            else:
                position += 1
        return minimized

    def _remove_unnecessary_rules(self, encoded, rules, approximation_mask):
        rules = list(rules)
        position = len(rules) - 1
        while position >= 0:
            other_rules = rules[:position] + rules[position + 1 :]
            if other_rules:
                coverage = np.zeros(len(encoded), dtype=bool)
                for rule in other_rules:
                    coverage |= self._rule_mask(encoded, rule)
                if np.all(coverage[approximation_mask]):
                    rules.pop(position)
            position -= 1
        return rules

    def _label_coverage(
        self,
        encoded,
        labels,
        label_to_coverage,
        row_groups,
        row_group_counts,
        only_certain=True,
        verbose=1,
    ):
        start_time = time.perf_counter()
        label_mask = labels == label_to_coverage
        duplicate_mask = row_group_counts[row_groups] > 1

        if only_certain:
            approximation_mask = label_mask & ~duplicate_mask
        else:
            target_groups = np.unique(row_groups[label_mask])
            approximation_mask = np.isin(row_groups, target_groups)

        if verbose == 2:
            rule_type = "certain" if only_certain else "uncertain"
            indexes = np.flatnonzero(approximation_mask).tolist()
            print(
                f"All objects to coverage with {rule_type} rules for "
                f"'{label_to_coverage}' class: {indexes}\n"
            )

        uncovered = approximation_mask.copy()
        rules = []
        iterations = 0
        global_counts = [
            np.bincount(encoded[:, column])
            for column in range(encoded.shape[1])
        ]

        while np.any(uncovered):
            iterations += 1
            current_rule = []
            current_mask = np.ones(len(encoded), dtype=bool)

            while np.any(current_mask & ~approximation_mask):
                candidates = []
                active = uncovered & current_mask

                for column in range(encoded.shape[1]):
                    values = encoded[active, column]
                    if len(values) == 0:
                        continue
                    counts = np.bincount(
                        values, minlength=len(global_counts[column])
                    )
                    for value_code in np.flatnonzero(counts):
                        descriptor = (column, int(value_code))
                        if descriptor not in current_rule:
                            candidates.append(
                                (
                                    -int(counts[value_code]),
                                    int(global_counts[column][value_code]),
                                    column,
                                    int(value_code),
                                )
                            )

                if not candidates:
                    raise RuntimeError(
                        "Cannot construct a rule for the selected approximation."
                    )

                _, _, column, value_code = min(candidates)
                current_rule.append((column, value_code))
                current_mask &= encoded[:, column] == value_code

                if verbose == 2:
                    descriptor = [
                        self.feature_names_[column],
                        value_code,
                        label_to_coverage,
                    ]
                    print(f"Iteration #{iterations}, selected: {descriptor}")

            current_rule = self._minimalize_rule(
                encoded, current_rule, approximation_mask
            )
            rules.append(current_rule)
            uncovered &= ~self._rule_mask(encoded, current_rule)

        rules = self._remove_unnecessary_rules(
            encoded, rules, approximation_mask
        )

        if verbose > 0:
            elapsed = round(time.perf_counter() - start_time, 2)
            print(f"Coveraged in {iterations + 1} iterations ({elapsed} s)")
        return rules

    def _decode_rule(self, rule, label):
        decoded = []
        for column, value_code in rule:
            value = self._code_to_value[column][value_code]
            if value == self._MISSING:
                value = "None"
            decoded.append([self.feature_names_[column], value, label])
        return decoded

    def fit(self, data, labels, only_certain=True, verbose=1):
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")

        original_labels = pd.Series(labels, copy=False).reset_index(drop=True)
        unique_labels = pd.unique(original_labels).tolist()
        label_counts = original_labels.value_counts(sort=False, dropna=False)
        original_positions = {
            label: position for position, label in enumerate(unique_labels)
        }
        self.label_counts_ranking = sorted(
            unique_labels,
            key=lambda label: (
                -int(label_counts.loc[label]),
                -original_positions[label],
            ),
        )

        _, encoded, labels_array = self._prepare_training_data(data, labels)
        if encoded.shape[1] == 0 and len(encoded):
            raise ValueError("Data must contain at least one attribute.")

        unique_labels = pd.unique(labels_array).tolist()
        self._label_to_index = {
            label: index for index, label in enumerate(self.label_counts_ranking)
        }

        row_groups, row_group_counts = self._row_groups(encoded)
        encoded_rules = []
        decoded_rules = []

        if verbose > 0:
            print("\nTrain process:")

        for index, label in enumerate(unique_labels):
            if verbose > 0:
                print(
                    f"\t{index + 1}/{len(unique_labels)} label ({label})",
                    end="\t",
                )
                sys.stdout.flush()

            label_rules = self._label_coverage(
                encoded,
                labels_array,
                label,
                row_groups,
                row_group_counts,
                only_certain=only_certain,
                verbose=verbose,
            )
            for rule in label_rules:
                encoded_rules.append((rule, label))
                decoded_rules.append(self._decode_rule(rule, label))

        if verbose > 0:
            print()

        self._encoded_rules = encoded_rules
        self.rules = decoded_rules
        return self.rules

    def _encode_prediction_data(self, data):
        if not self.feature_names_:
            raise RuntimeError("The classifier must be fitted before prediction.")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.feature_names_)

        missing_columns = [
            name for name in self.feature_names_ if name not in data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing attributes: {missing_columns}")

        data = data.loc[:, self.feature_names_].fillna(self._MISSING)
        columns = []
        for column, name in enumerate(self.feature_names_):
            codes = data[name].map(self._value_to_code[column]).fillna(-1)
            columns.append(codes.to_numpy(dtype=np.int64))
        return np.column_stack(columns)

    def predict(self, data, verbose=1):
        encoded = self._encode_prediction_data(data)
        class_count = len(self.label_counts_ranking)
        votes = np.zeros((len(encoded), class_count), dtype=np.int32)

        for rule, label in self._encoded_rules:
            votes[self._rule_mask(encoded, rule), self._label_to_index[label]] += 1

        max_votes = votes.max(axis=1)
        winning_indexes = np.argmax(votes, axis=1)
        ranking = np.asarray(self.label_counts_ranking, dtype=object)
        predictions = ranking[winning_indexes].tolist()
        non_predicted_counter = int(np.count_nonzero(max_votes == 0))
        tied_classes = np.count_nonzero(votes == max_votes[:, None], axis=1)
        conflicts_counter = int(
            np.count_nonzero((max_votes > 0) & (tied_classes > 1))
        )

        if verbose > 0:
            print(f"\nDuring the prediction, {conflicts_counter} conflicts occurred.")
            print(
                "The prediction process included "
                f"{non_predicted_counter} objects whose class could not be predicted.\n"
            )
        return predictions

    def _predict_object_class(self, object, rules=None, verbose=1):
        frame = pd.DataFrame([object])
        encoded = self._encode_prediction_data(frame)
        matched_labels = [
            label
            for rule, label in self._encoded_rules
            if self._rule_mask(encoded, rule)[0]
        ]
        if not matched_labels:
            return self.label_counts_ranking[0], False, True

        counts = {
            label: matched_labels.count(label) for label in set(matched_labels)
        }
        maximum = max(counts.values())
        winners = [
            label for label in self.label_counts_ranking
            if counts.get(label) == maximum
        ]
        return winners[0], len(winners) > 1, False

    def print_rules(self):
        for counter, (rule, (_, label)) in enumerate(
            zip(self.rules, self._encoded_rules), start=1
        ):
            conditions = " && ".join(
                f"'{attribute}={value}'" for attribute, value, _ in rule
            )
            if not conditions:
                conditions = "TRUE"
            print(f"{counter}) {conditions} => 'label={label}'")

    def evaluate(self, data, labels, verbose=1):
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")
        predictions = self.predict(data, verbose=verbose)
        accuracy = float(np.mean(np.asarray(labels) == np.asarray(predictions)))
        if verbose > 0:
            print(f"Accuracy of evaluate: {accuracy}")
        return "accuracy", accuracy
