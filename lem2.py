import pandas as pd
import sys

class LEM2:
    """
    LEM2 is a class representing a rule-based inference classifier that implements the LEM2 algorithm.
    
    Attributes:
        rules: List of all inference rules.
        label_counts_ranking: Class ranking based on the frequency of class occurrence in the training data.
    """
    
    def __init__(self) -> None:
        
        """
        Initializes the LEM2 object.
        """
        
        self.rules = []
        self.label_counts_ranking = []

    # -------------------------------------------------------------------

    def check_list_of_descriptors_is_enough(self, data: pd.DataFrame, T: list, B:list ) -> bool:
        
        """
        Checks whether the list of descriptors is sufficient (certain or uncertain) on the data set and the selected decision class approximation.
        
        Params:
            data: Information system.
            T: List of descriptors.
            B: Selected approximation of the decision class.
            
        Return:
            bool: Information whether the rule is sufficient (certain or uncertain).
        """
        
        objects_to_check = list(set(data.index) - set(B))
        
        for index in objects_to_check:
            rule_recognize = True

            for descriptor in T:
                if data.iloc[index][descriptor[0]] != descriptor[1]:
                    rule_recognize = False
                    break
            
            if rule_recognize: return False
            
        
        return True

    # -------------------------------------------------------------------

    def objects_recognized_by_rule(self, data: pd.DataFrame, rule: list) -> list:
        
        """
        Returns a list of indexes of objects recognized by a given rule from the information system.
        
        Params:
            data: Information system.
            rule: Rule.
            
        Return:
            list: List of indexes of objects which that rule recognize.
        """
        
        recognized_objects = []
        
        for index in range(len(data)):
            is_recognized = True
            
            for descriptor in rule:
                if data[descriptor[0]][index] != descriptor[1]:
                    is_recognized = False
                    break
            
            if is_recognized: recognized_objects.append(index)
        
        return recognized_objects
            
    # -------------------------------------------------------------------

    def minimalize_rule(self, data: pd.DataFrame, rule: list, B: list) -> list:
        
        """
        Minimizes the rule by removing unnecessary descriptors.
        
        Params:
            data: Information system.
            rule: Rule.
            B: Selected approximation of the decision class.
            
        Return:
            list: Minimalized rule.
        """
        
        filter_for_rule = []
        
        if len(rule) == 1: return rule
        
        
        temp = rule.copy()
        
            
        # Check that descriptors are necessary
        for descriptor in rule:
            temp.remove(descriptor)
            
            verdict = self.check_list_of_descriptors_is_enough(data, temp, B)
            
            if verdict: filter_for_rule.append(False)
            else: filter_for_rule.append(True)
            
        new_rule = [rule[i] for i in range(len(rule)) if filter_for_rule[i]]
        
        return new_rule

    # -------------------------------------------------------------------

    def label_coverage(self, data: pd.DataFrame, labels: list, label_to_coverage, only_certain=True, verbose=1) -> list:
        
        """
        Generates minimum coverage of the selected decision class using simple and minimal rules.
        
        Params:
            data: Information system.
            labels: Labels of objects.
            label_to_coverage: Selected decision class to cover.
            only_certain: Flag to indicate whether the algorithm should generate only certain rules.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            list: Minimal coverage of selected decision class.
            
        Raise:
            ValueError: Data and labels must have same length.
        """
    
        
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")

        # Remove nonunique rows with decisions
        data['label'] = labels
        data = data.drop_duplicates().reset_index(drop=True)
        
        if verbose == 2: print(f"\nOptimalized data:\n{data}")
        
        labels = data['label']
        data = data.drop('label', axis=1)
        
        rules = []

        B = []

        # Get objects from selected class
        index_to_check = [index for index in range(len(data)) if labels[index] == label_to_coverage]

        # Based on selected rule type define B
        if only_certain:
            for index in index_to_check:
                if not data.duplicated(keep=False).iloc[index]:
                    B.append(index)
                    
            if verbose == 2: print(f"All objects to coverage with certain rules for '{label_to_coverage}' class: {B}")

        else:
            for index in index_to_check:
                B.append(index)
                if data.duplicated(keep=False).iloc[index]:
                    identical_indexes = data[(data == data.iloc[index]).all(axis=1)].index
                    B.extend([i for i in identical_indexes if i != index])
                
            if verbose == 2: print(f"All objects to coverage with uncertain rules for '{label_to_coverage}' class: {B}")
        
        
        if verbose == 2: print()
        counter = 1
                    
        G = B.copy()
        T = []
        
        while len(G) != 0:
            
            if verbose == 2: print(f"Iteration #{counter}")
            
            descriptors = []
            
            # Get all descriptors, which we can get from objects from G
            for attribute in data:
                for object_index in G:
                    descriptor = [attribute, data[attribute][object_index], label_to_coverage]
                    if not descriptor in T:
                        descriptors.append( [descriptor] )
            
            
            criterions_counts = []
            
            # To each descriptor add it value for first criterion
            for descriptor in descriptors:
                temp = descriptor.copy()
                temp.append(descriptors.count(descriptor))
                
                if not temp in criterions_counts:
                    criterions_counts.append(temp)
            
            
            # To each descriptor add it value for second criterion
            for descriptor in criterions_counts:
                value = int((data[descriptor[0][0]] == descriptor[0][1]).sum())
                descriptor.append(value)
                
            if verbose == 2: print("Descriptors: [[attribute, value, class], first_criterion, second_criterion]")
            if verbose == 2: print(criterions_counts)
            
            t = ['-', 0]
            
            # Calculate values for attributes for best descriptor
            max_first_criterion_value = max(descriptor[1] for descriptor in criterions_counts)
            
            indexes_to_get_from_first_criterion = [i for i in range(len(criterions_counts)) if criterions_counts[i][1] == max_first_criterion_value]
            
            min_second_criterion_value = min(criterions_counts[index][2] for index in indexes_to_get_from_first_criterion)
            
            if verbose == 2: print(f"Max first criterion: {max_first_criterion_value}, min second criterion: {min_second_criterion_value}")
            
            # Select next descriptor (t) and add it to T
            for descriptor in criterions_counts:
                if descriptor[1] == max_first_criterion_value and descriptor[2] == min_second_criterion_value:
                    t = descriptor[0]
                    if verbose == 2:  print(f"t = {t}")
                    T.append(t)
                    break
                    
            if verbose == 2: print(f"T = {T}")
            
            # Check is rule enough?
            if verbose == 2:  print(f"Is rule enough? {self.check_list_of_descriptors_is_enough(data, T, B)}")
            
            if self.check_list_of_descriptors_is_enough(data, T, B):
                
                # Minimalize enough rule
                T = self.minimalize_rule(data, T, B)
                
                if verbose == 2: print(f"Rule after minimalization: {T}")
                
                # Add new rule
                rules.append(T)
                T = []
                
                # Check object no coveraged in B
                filter_for_B = []
                
                for index in B:
                    is_coverage = False
                    
                    for rule in rules:
                        rule_coverage_object = True
                        
                        for descriptor in rule:
                            if data[descriptor[0]][index] != descriptor[1]:
                                rule_coverage_object = False
                                break
                        
                        if rule_coverage_object:
                            filter_for_B.append(False)
                            is_coverage = True
                            break
                    
                    if not is_coverage: filter_for_B.append(True)
                    
                # Define G to next iteration
                temp = [B[index] for index in range(len(B)) if filter_for_B[index]]
                G = temp.copy()
                
                if verbose == 2: print(f"G after new rule: {G}")
                        
                
            else:
                # If rule not enough then remove from G objects that we cannot coverage now
                filter_for_G = []
                
                for index in G:
                    object_is_correct = True
                    
                    for descriptor in T:
                        if data[descriptor[0]][index] != descriptor[1]:
                            object_is_correct = False
                    
                    filter_for_G.append(object_is_correct)
                    
                # Define G to next iteration
                temp = [G[index] for index in range(len(G)) if filter_for_G[index]]
                G = temp.copy()
                if verbose == 2: print(f"New G after reduction: {G}")
                
            counter += 1
            if verbose == 2: print()
            
        # Remove unnecessary rules
        self.remove_unnecessary_rules(data, rules)
        
        if verbose > 0: print(f"Coveraged in {counter} iterations")
                        
        # Return rules for selected label
        return rules
                    
    # -------------------------------------------------------------------

    def remove_unnecessary_rules(self, data: pd.DataFrame, rules: list) -> list:
        
        """
        Removes unnecessary rules.
        
        Params:
            data: Information system.
            rules: List of rules.
            
        Return:
            list: List of necessary rules.
        """
        
        coveraged_objects = []
        for rule in rules: coveraged_objects.append(self.objects_recognized_by_rule(data, rule))
        filter_for_rules = []
        
        for rule_coverage in coveraged_objects:
            has_unique_object = False
            
            for object in rule_coverage:
                
                temp = coveraged_objects.copy()
                temp.remove(rule_coverage)
                
                for temp_coverage in temp:
                    if object in temp_coverage:
                        break
                        
                    has_unique_object = True
            
            if has_unique_object: filter_for_rules.append(True)
            else: filter_for_rules.append(False)
            
        return [rules[i] for i in range(len(rules)) if filter_for_rules[i]]

    # -------------------------------------------------------------------

    def print_rules(self) -> None:
    
        """
        Prints all inference rules fitted by the algorithm.
        """
        
        counter = 1
        for rule in self.rules:
            print(f"{counter}) ", end="")
            
            for index in range(len(rule)):
                            
                print(f"'{rule[index][0]}={rule[index][1]}' ", end="")
                
                if index != len(rule)-1:
                    print("&& ", end="")
            
            print(f"=> 'label={rule[0][2]}'")
            counter += 1
            
    # -------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, labels: list, only_certain=True, verbose=1) -> list:
        
        """
        Learns classifier object, i.e. generates rules.
        
        Params:
            data: Information system.
            labels: Labels of objects.
            only_certain: A flag to indicate whether the algorithm should generate only certain rules.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            list: List of rules, which will be saved in classifier.
            
        Raise:
            ValueError: Data and labels must have same length.
        """
        
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")
        
        all_labels = labels.unique()
        
        labels_with_counts = dict()
        
        
        # Create ranking for labels (in proediction conflict will be selected most common class)
        for label in all_labels:
            number = list(labels).count(label)
            labels_with_counts[label] = number
            
        sorted_labels_with_counts = sorted(labels_with_counts.items(), key=lambda x: x[1])
        
        for key, value in sorted_labels_with_counts:
            self.label_counts_ranking.insert(0, key)
            
        if verbose == 2: print(f"Labels ranking: {self.label_counts_ranking}")
            
        # Training process
        rules = []
        
        if verbose > 0:
                print(f"\nTrain process:")
        
        for index in range(len(all_labels)):
            
            if verbose > 0:
                print(f"\t{index+1}/{len(all_labels)} label ({all_labels[index]})", end="\t")
                sys.stdout.flush()

                
            rule_for_label = self.label_coverage(data, labels, label_to_coverage=all_labels[index], only_certain=only_certain, verbose=verbose)
            rules.extend(rule_for_label)
            
            
            
        if verbose > 0:
                print()
        
        self.rules = rules
        
        return rules
    
    # -------------------------------------------------------------------
    
    def predict_object_class(self, object: dict, rules: list, verbose=1):
        
        """
        Predicts a class on the selected object. In case of conflict, it selects the first decision class found.
        
        Params:
            object: Object passed as a dictionary that must contain all fields from the training set.
            rules: List of rules.
            verbose: Mode of generating messages by the predicting process.
            
        Return:
            Decision class or None if no match is found.
            bool: Flag indicating that a conflict occurred while predicting the class for this object.
        """
        
        predicted_classes = []
        
        # Get all predicted classes for object
        for rule in rules:
            is_recognized_by_rule = True
            
            for descriptor in rule:
                if object[descriptor[0]] != descriptor[1]:
                    is_recognized_by_rule = False
                    
            if is_recognized_by_rule: predicted_classes.append(rule[0][2])
            
        if verbose == 2: print(f"Classes for object {object}: {predicted_classes}")
        
        # If no rules return None
        if len(predicted_classes) == 0: 
            if verbose == 2: print(f"Class for {object}: {None}")
            return None, False
        
        # Search most common decision  
        set_of_classes = set(predicted_classes)
        counts = []
        
        for elem in set_of_classes:
            counts.append(predicted_classes.count(elem))
        
                         
        max_count = max(counts)
        
        is_conflict = False
        if (counts.count(max_count) > 1):
            is_conflict = True
            
        conflicted_classes = []
        
        
        for i in range(len(counts)):
            
            if not is_conflict:
                if max_count == counts[i]:
                    
                    if verbose == 2: print(f"Class for {object}: {list(set_of_classes)[i]}")
                    
                    return list(set_of_classes)[i], is_conflict
                
            if is_conflict:
                if max_count == counts[i]:
                    conflicted_classes.append(list(set_of_classes)[i])
        
        
        # If conflict, select most popular label from train data
        positions_in_ranking = []
        for label in conflicted_classes: positions_in_ranking.append(self.label_counts_ranking.index(label))
        
        best_position = min(positions_in_ranking)
        
        for i in range(len(positions_in_ranking)):
            if positions_in_ranking[i] == best_position:
                
                if verbose == 2: print(f"Class for {object}: {conflicted_classes[i]}")
                    
                return conflicted_classes[i], is_conflict
        
    # -------------------------------------------------------------------
    
    def predict(self, data: pd.DataFrame, verbose=1) -> list:
        
        """
        Predicts classes for all objects in the passed data frame.
        
        Params:
            data: Information system.
            verbose: Mode of generating messages by the predicting process.
            
        Return:
            list: List of predicted classes.
        """
            
        classes = []
        conflicts_counter = 0
        none_counter = 0
        
        if verbose == 2: print(f"Objects to predict: {len(data)}")
        
        for i in range(len(data)):
            predict_class, is_conflict = self.predict_object_class(data.iloc[i].to_dict(), self.rules, verbose=verbose)
            classes.append(predict_class)
            
            if is_conflict: conflicts_counter += 1
            if predict_class == None: none_counter += 1
            
            if verbose == 2: print(f"\t{i+1}/{len(data)} classes predicted")
            
        
        if verbose > 0: print(f"During the prediction, {conflicts_counter} conflicts occurred.")
        if verbose > 0: print(f"The prediction process included {none_counter} objects whose class could not be predicted.")
        
        if verbose == 2: print(f"All predicted classes: {classes}")
        
        if verbose > 0: print()
        
        return classes       
        
    # --------------------------------------------------------------------
    
    def evaluate(self, data: pd.DataFrame, labels: list, verbose=1) -> tuple:
        
        """
        Calculates the accuracy metric for the prediction process.
        
        Params:
            data: Information system.
            labels: Labels of objects.
            verbose: Mode of generating messages by the evaluating process.
            
        Return:
            tuple: Tuple of metrics.
            
        Raise:
            ValueError: Data and labels must have same length.
        """
        
        if len(data) != len(labels):
            raise ValueError("Data and labels must have same length.")
                
        preds = self.predict(data, verbose=verbose)
        
        errors_counter = sum([labels[i] != preds[i] for i in range(len(preds))])
        
        accuracy = float(len(labels)-errors_counter)/(len(labels))
        
        if verbose > 0: print(f"Accuracy of evaluate: {accuracy}")
        
        return ('accuracy', accuracy)