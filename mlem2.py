import sys
import time

import numpy as np
import pandas as pd


class MLEM2:
    """
    Rule classifier implementing the MLEM2 algorithm.

    Symbolic attributes produce descriptors ``a = v``. Numerical attributes
    produce descriptors ``a < v`` and ``a >= v``, where ``v`` is the midpoint
    between two consecutive distinct values observed during training.

    Attributes:
        rules: Induced decision rules. Every descriptor has the form
            [attribute, operator, value, decision].
        label_counts_ranking: Decision classes ordered by their frequency.
        cut_points_: Cut points generated for numerical attributes.
    """

    _MISSING = "__MLEM2_MISSING__"

    def __init__(self) -> None:
        self.rules = []
        self.label_counts_ranking = []
        self.cut_points_ = {}
        self.feature_names_ = []
        self.numeric_attributes_ = []

        self._descriptors = []
        self._descriptor_masks = np.empty((0, 0), dtype=bool)
        self._internal_rules = []
        self._label_to_index = {}
        self._symbolic_values = {}
        self._training_row_groups = np.empty(0, dtype=np.int64)
        self._training_group_counts = np.empty(0, dtype=np.int64)

    @staticmethod
    def _validate_lengths(data, labels):
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")

    def _prepare_training_data(self, data, labels):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        self._validate_lengths(data, labels)
        data = data.reset_index(drop=True).copy()
        labels = pd.Series(labels, copy=False).reset_index(drop=True)

        if len(data) == 0:
            raise ValueError("Training data must contain at least one object.")
        if data.shape[1] == 0:
            raise ValueError("Data must contain at least one attribute.")
        if data.columns.duplicated().any():
            raise ValueError("Attribute names must be unique.")

        self.feature_names_ = list(data.columns)
        self.numeric_attributes_ = [
            name
            for name in self.feature_names_
            if pd.api.types.is_numeric_dtype(data[name].dtype)
            and not pd.api.types.is_bool_dtype(data[name].dtype)
        ]

        normalized = data.copy()
        for name in self.feature_names_:
            if name in self.numeric_attributes_:
                normalized[name] = pd.to_numeric(
                    normalized[name], errors="raise"
                ).astype(float)
            else:
                normalized[name] = normalized[name].astype(object)
                normalized[name] = normalized[name].where(
                    normalized[name].notna(), self._MISSING
                )

        combined = normalized.copy()
        combined["__mlem2_label__"] = labels.to_numpy()
        combined = combined.drop_duplicates().reset_index(drop=True)
        labels = combined.pop("__mlem2_label__").to_numpy()

        return normalized, combined, labels

    def _create_row_groups(self, data):
        encoded_columns = []
        for name in self.feature_names_:
            column = data[name]
            if name in self.numeric_attributes_:
                missing = column.isna().to_numpy()
                values = column.fillna(0.0).to_numpy(dtype=float)
                pairs = pd.Series(
                    list(zip(missing.tolist(), values.tolist())), dtype=object
                )
                codes, _ = pd.factorize(pairs, sort=False)
            else:
                codes, _ = pd.factorize(column, sort=False)
            encoded_columns.append(codes.astype(np.int64, copy=False))

        encoded = np.column_stack(encoded_columns)
        _, row_groups, group_counts = np.unique(
            encoded, axis=0, return_inverse=True, return_counts=True
        )
        return row_groups, group_counts

    def _generate_descriptors(self, data):
        descriptors = []
        masks = []
        self.cut_points_ = {}
        self._symbolic_values = {}

        for column_index, name in enumerate(self.feature_names_):
            column = data[name]

            if name in self.numeric_attributes_:
                values = np.sort(column.dropna().unique().astype(float))
                if len(values) > 1:
                    cuts = values[:-1] + (values[1:] - values[:-1]) / 2.0
                    cuts = np.round(cuts, decimals=12)
                else:
                    cuts = np.empty(0, dtype=float)

                self.cut_points_[name] = cuts.tolist()
                numeric_values = column.to_numpy(dtype=float)

                for cut in cuts:
                    descriptors.append((column_index, "<", float(cut)))
                    masks.append(numeric_values < cut)
                    descriptors.append((column_index, ">=", float(cut)))
                    masks.append(numeric_values >= cut)

                if column.isna().any():
                    descriptors.append((column_index, "=", self._MISSING))
                    masks.append(column.isna().to_numpy())
            else:
                values = pd.unique(column).tolist()
                self._symbolic_values[name] = values
                object_values = column.to_numpy(dtype=object)
                for value in values:
                    descriptors.append((column_index, "=", value))
                    masks.append(object_values == value)

        if masks:
            descriptor_masks = np.vstack(masks).astype(bool, copy=False)
        else:
            descriptor_masks = np.empty((0, len(data)), dtype=bool)

        self._descriptors = descriptors
        self._descriptor_masks = descriptor_masks

    def _rule_mask(self, rule):
        if not rule:
            return np.ones(self._descriptor_masks.shape[1], dtype=bool)
        return np.all(self._descriptor_masks[np.asarray(rule)], axis=0)

    def _rule_is_sufficient(self, rule, approximation_mask):
        rule_mask = self._rule_mask(rule)
        return np.any(rule_mask) and not np.any(rule_mask & ~approximation_mask)

    def _minimalize_rule(self, rule, approximation_mask):
        minimized = list(rule)
        position = 0

        while position < len(minimized):
            candidate = minimized[:position] + minimized[position + 1 :]
            if self._rule_is_sufficient(candidate, approximation_mask):
                minimized = candidate
            else:
                position += 1

        return minimized

    def _remove_unnecessary_rules(self, rules, approximation_mask):
        rules = list(rules)
        position = len(rules) - 1

        while position >= 0:
            remaining = rules[:position] + rules[position + 1 :]
            coverage = np.zeros(len(approximation_mask), dtype=bool)
            for rule in remaining:
                coverage |= self._rule_mask(rule)

            if remaining and np.all(coverage[approximation_mask]):
                rules.pop(position)
            position -= 1

        return rules

    def _approximation_mask(self, labels, label, only_certain):
        label_mask = labels == label
        groups = self._training_row_groups

        target_count = np.bincount(
            groups,
            weights=label_mask.astype(np.int64),
            minlength=len(self._training_group_counts),
        )

        if only_certain:
            accepted_groups = target_count == self._training_group_counts
        else:
            accepted_groups = target_count > 0

        return accepted_groups[groups]

    def _format_descriptor(self, descriptor_index, label=None):
        column, operator, value = self._descriptors[descriptor_index]
        if value == self._MISSING:
            value = "None"
        descriptor = [self.feature_names_[column], operator, value]
        if label is not None:
            descriptor.append(label)
        return descriptor

    def _label_coverage(
        self, labels, label_to_coverage, only_certain=True, verbose=1
    ):
        start_time = time.perf_counter()
        approximation_mask = self._approximation_mask(
            labels, label_to_coverage, only_certain
        )

        if not np.any(approximation_mask):
            if verbose > 0:
                print("No objects in the selected approximation (0.0 s)")
            return []

        if verbose == 2:
            rule_type = "certain" if only_certain else "possible"
            indexes = (np.flatnonzero(approximation_mask) + 1).tolist()
            print(
                f"Objects covered by {rule_type} rules for "
                f"'{label_to_coverage}': {indexes}"
            )

        descriptor_sizes = self._descriptor_masks.sum(axis=1)
        uncovered = approximation_mask.copy()
        rules = []
        iterations = 0

        while np.any(uncovered):
            iterations += 1
            current_rule = []
            current_mask = np.ones(len(labels), dtype=bool)
            G = uncovered.copy()

            while np.any(current_mask & ~approximation_mask):
                overlaps = np.count_nonzero(
                    self._descriptor_masks & G, axis=1
                )

                if current_rule:
                    overlaps[np.asarray(current_rule)] = 0

                usable = np.flatnonzero(overlaps > 0)
                if len(usable) == 0:
                    raise RuntimeError(
                        "Cannot construct a rule for the selected approximation."
                    )

                maximum_overlap = overlaps[usable].max()
                best = usable[overlaps[usable] == maximum_overlap]
                minimum_block = descriptor_sizes[best].min()
                best = best[descriptor_sizes[best] == minimum_block]
                selected = int(best[0])

                current_rule.append(selected)
                current_mask &= self._descriptor_masks[selected]
                G = uncovered & current_mask

                if verbose == 2:
                    print(
                        "Selected descriptor:",
                        self._format_descriptor(
                            selected, label_to_coverage
                        ),
                        f"|[t] intersect G|={maximum_overlap}",
                        f"|[t]|={descriptor_sizes[selected]}",
                    )

            current_rule = self._minimalize_rule(
                current_rule, approximation_mask
            )
            rules.append(current_rule)
            uncovered &= ~self._rule_mask(current_rule)

            if verbose == 2:
                readable = [
                    self._format_descriptor(index, label_to_coverage)
                    for index in current_rule
                ]
                print(f"Rule #{iterations}: {readable}")
                print(
                    "Uncovered objects:",
                    (np.flatnonzero(uncovered) + 1).tolist(),
                )

        rules = self._remove_unnecessary_rules(rules, approximation_mask)

        if verbose > 0:
            elapsed = round(time.perf_counter() - start_time, 2)
            print(f"Covered in {iterations} iterations ({elapsed} s)")

        return rules

    def fit(self, data, labels, only_certain=True, verbose=1):
        """
        Induces MLEM2 decision rules.

        Args:
            data: Data frame containing condition attributes.
            labels: Decision values.
            only_certain: Generate rules from lower approximations when True,
                or from upper approximations when False.
            verbose: 0 disables messages, 1 prints progress, 2 prints details.

        Returns:
            List of induced rules.
        """
        self._validate_lengths(data, labels)
        original_labels = pd.Series(labels, copy=False).reset_index(drop=True)
        unique_labels = pd.unique(original_labels).tolist()
        counts = original_labels.value_counts(sort=False, dropna=False)
        positions = {
            label: position for position, label in enumerate(unique_labels)
        }
        self.label_counts_ranking = sorted(
            unique_labels,
            key=lambda label: (
                -int(counts.loc[label]),
                -positions[label],
            ),
        )
        self._label_to_index = {
            label: index for index, label in enumerate(self.label_counts_ranking)
        }

        _, training_data, training_labels = self._prepare_training_data(
            data, labels
        )
        self._training_row_groups, self._training_group_counts = (
            self._create_row_groups(training_data)
        )
        self._generate_descriptors(training_data)

        if len(self._descriptors) == 0:
            raise ValueError(
                "No descriptors can be generated from the training data."
            )

        rules = []
        readable_rules = []
        unique_training_labels = pd.unique(training_labels).tolist()

        if verbose > 0:
            print("\nTrain process:")

        for position, label in enumerate(unique_training_labels, start=1):
            if verbose > 0:
                print(
                    f"\t{position}/{len(unique_training_labels)} "
                    f"label ({label})",
                    end="\t",
                )
                sys.stdout.flush()

            label_rules = self._label_coverage(
                training_labels,
                label,
                only_certain=only_certain,
                verbose=verbose,
            )
            for rule in label_rules:
                rules.append((rule, label))
                readable_rules.append(
                    [
                        self._format_descriptor(index, label)
                        for index in rule
                    ]
                )

        if verbose > 0:
            print()

        self._internal_rules = rules
        self.rules = readable_rules
        return self.rules

    def _prepare_prediction_data(self, data):
        if not self.feature_names_:
            raise RuntimeError("The classifier must be fitted before prediction.")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.feature_names_)

        missing = [
            name for name in self.feature_names_ if name not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing attributes: {missing}")

        data = data.loc[:, self.feature_names_].reset_index(drop=True).copy()
        for name in self.feature_names_:
            if name in self.numeric_attributes_:
                data[name] = pd.to_numeric(data[name], errors="raise").astype(
                    float
                )
            else:
                data[name] = data[name].astype(object)
                data[name] = data[name].where(
                    data[name].notna(), self._MISSING
                )
        return data

    def _descriptor_mask_for_data(self, data, descriptor_index):
        column, operator, value = self._descriptors[descriptor_index]
        name = self.feature_names_[column]
        column_values = data[name].to_numpy()

        if operator == "<":
            return column_values < value
        if operator == ">=":
            return column_values >= value
        return column_values == value

    def _prediction_rule_mask(self, data, rule):
        mask = np.ones(len(data), dtype=bool)
        for descriptor_index in rule:
            mask &= self._descriptor_mask_for_data(data, descriptor_index)
        return mask

    def predict(self, data, verbose=1):
        """Predicts decision classes for objects from ``data``."""
        data = self._prepare_prediction_data(data)
        if len(data) == 0:
            return []

        votes = np.zeros(
            (len(data), len(self.label_counts_ranking)), dtype=np.int32
        )

        for rule, label in self._internal_rules:
            mask = self._prediction_rule_mask(data, rule)
            votes[mask, self._label_to_index[label]] += 1

        maximum_votes = votes.max(axis=1)
        winning_indexes = np.argmax(votes, axis=1)
        ranking = np.asarray(self.label_counts_ranking, dtype=object)
        predictions = ranking[winning_indexes].tolist()

        unresolved = maximum_votes == 0
        tied = np.count_nonzero(
            votes == maximum_votes[:, None], axis=1
        ) > 1
        conflict_count = int(np.count_nonzero(tied & ~unresolved))
        unresolved_count = int(np.count_nonzero(unresolved))

        if verbose > 0:
            print(f"\nDuring the prediction, {conflict_count} conflicts occurred.")
            print(
                "The prediction process included "
                f"{unresolved_count} objects whose class could not be predicted."
            )
            print()

        return predictions

    def _predict_object_class(self, object, rules=None, verbose=1):
        """Predicts one object and returns class and diagnostic flags."""
        data = self._prepare_prediction_data(pd.DataFrame([object]))
        matched_labels = []

        for rule, label in self._internal_rules:
            if self._prediction_rule_mask(data, rule)[0]:
                matched_labels.append(label)

        if not matched_labels:
            return self.label_counts_ranking[0], False, True

        counts = {
            label: matched_labels.count(label) for label in set(matched_labels)
        }
        maximum = max(counts.values())
        winners = [
            label
            for label in self.label_counts_ranking
            if counts.get(label) == maximum
        ]

        if verbose == 2:
            print(f"Classes for object {object}: {matched_labels}")
            print(f"Class for object: {winners[0]}")

        return winners[0], len(winners) > 1, False

    def print_rules(self):
        """Prints all induced decision rules."""
        for counter, (rule, label) in enumerate(
            self._internal_rules, start=1
        ):
            conditions = []
            for descriptor_index in rule:
                attribute, operator, value, _ = self._format_descriptor(
                    descriptor_index, label
                )
                conditions.append(f"'{attribute} {operator} {value}'")

            antecedent = " && ".join(conditions) if conditions else "TRUE"
            print(f"{counter}) {antecedent} => 'label={label}'")

    def evaluate(self, data, labels, verbose=1):
        """Calculates classification accuracy."""
        self._validate_lengths(data, labels)
        predictions = self.predict(data, verbose=verbose)
        accuracy = float(
            np.mean(np.asarray(labels) == np.asarray(predictions))
        )

        if verbose > 0:
            print(f"Accuracy of evaluate: {accuracy}")

        return "accuracy", accuracy
