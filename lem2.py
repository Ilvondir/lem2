class LEM2:
    
    def __init__(self) -> None:
        self.rules = []

    # -------------------------------------------------------------------

    def check_rule_is_enough(self, data, T, B) -> bool:
        
        """
        Checks whether the rule is sufficient (certain or uncertain) on the data set and the selected decision class approximation.
        
        Params:
            data: Decision system
            T: Rule
            B: Selected approximation of the decision class
            
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

    def objects_recognized_by_rule(self, data, rule):
        
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

    def minimalize_rule(self, data, rule, B):
        filter_for_rule = []
        
        if len(rule) == 1: return rule
        
            
        # Check that descriptors are necessary
        for descriptor in rule:
            temp = rule.copy()
            temp.remove(descriptor)
            
            verdict = self.check_rule_is_enough(data, temp, B)
            
            if verdict: filter_for_rule.append(False)
            else: filter_for_rule.append(True)
            
        new_rule = [rule[i] for i in range(len(rule)) if filter_for_rule[i]]
        
        # print(f"Minimalized rule: {new_rule}")
        
        return new_rule

    # -------------------------------------------------------------------

    def label_coverage(self, data, labels, label_to_coverage, only_certain=True, verbose=1):

        # Remove nonunique rows with decisions
        data['label'] = labels
        data = data.drop_duplicates().reset_index().drop("index", axis=1)
        labels = data['label']
        data = data.drop('label', axis=1)
        
        
        
        if verbose == 2: print(f"Optimalized data:\n{data}")
        
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
            if verbose == 2:  print(f"Is rule enough? {self.check_rule_is_enough(data, T, B)}")
            
            if self.check_rule_is_enough(data, T, B):
                
                # Minimalize enough rule
                T = self.minimalize_rule(data, T, B)
                
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
        
                
        # Return rules for selected label
        return rules
                    
    # -------------------------------------------------------------------

    def remove_unnecessary_rules(self, data, rules):
        
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

    def print_rules(self):
        
        """
        Printing all inference rules fitted by the algorithm.
        """
        
        for rule in self.rules:
            for index in range(len(rule)):
                
                print(f"'{rule[index][0]}={rule[index][1]}' ", end="")
                
                if index != len(rule)-1:
                    print("&& ", end="")
            
            print(f"=> 'label={rule[0][2]}'")
            
    # -------------------------------------------------------------------

    def fit(self, data, labels, only_certain=True, verbose=1):
        
        """
        The learning process, i.e. generating minimal coverage of rules for each decision class.
        
        Params:
            data: Information system.
            labels: Labels of objects
            only_certain: A flag to indicate whether the algorithm should generate only certain rules.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            list: List of rules, which will be saved in classifier.
        """
        
        all_labels = labels.unique()
        rules = []
        
        if verbose > 0:
                print(f"\nFit process:")
        
        for index in range(len(all_labels)):
            rule_for_label = self.label_coverage(data, labels, label_to_coverage=all_labels[index], only_certain=only_certain, verbose=verbose)
            rules.extend(rule_for_label)
            
            if verbose > 0:
                print(f"\t{index+1}/{len(all_labels)} labels coveraged")
            
        if verbose > 0:
                print()
        
        self.rules = rules
        
        return rules