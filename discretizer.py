import pandas as pd

class Discretizer:
    
    """
    Discretizer is a class representing a discretizer that uses an unsupervised approach.
    
    Attributes:
        cuts: Dictionary storing all learned cuts on all data columns.
    """
    
    def __init__(self):
        
        """
        Initializes the Discretezer object.
        """
        
        self.cuts = dict()
        
    
    def fit(self, data: pd.DataFrame, attributes: list, number_of_output_values: int, verbose=0) -> None:
        
        """
        Fits discretizer, i.e. generates cuts that the discretizer object will remember.
        
        Params:
            data: Information system.
            attributes: List of attributes to discretize.
            number_of_output_value: The number of values the attribute can has after discretization.
            verbose: Mode of generating messages by the learning process.
        
        Raise:
            ValueError: Number of output values must be greater than 0.
        """
        
        if number_of_output_values < 1:
            raise ValueError("Number of output values must be greater than 0.")
        
        for column in attributes:
        
            min_value = min(data[column])
            max_value = max(data[column])
            
            step = (max_value - min_value) / number_of_output_values
            cuts = []
        
            
            for i in range(number_of_output_values+1):
                cuts.append(min_value + step*i)
                
            self.cuts[column] = cuts
            
        if verbose > 0: print(f"All learned cuts: {self.cuts}")
        
            
    
    def fit_discretize(self, data: pd.DataFrame, attributes: list, number_of_output_values: int, verbose=0) -> pd.DataFrame:
        
        """
        Fits discretizer and discretizes selected attributes.
        
        Params:
            data: Information system.
            attributes: List of attributes to discretize.
            number_of_output_value: The number of values the attribute can has after discretization.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            DataFrame: Discretized data.
        
        Raise:
            ValueError: Number of output values must be greater than 0.
        """
        
        if number_of_output_values < 1:
            raise ValueError("Number of output values must be greater than 0.")
        
        my_data = data.copy()
        
        for column in attributes:
        
            min_value = min(data[column])
            max_value = max(data[column])
            
            step = (max_value - min_value) / number_of_output_values
            cuts = []
        
            
            for i in range(number_of_output_values+1):
                cuts.append(min_value + step*i)
                
            self.cuts[column] = cuts
                        
            new_values = []
            
            for i in data.index:
                is_discretized = False
                
                for j in range(number_of_output_values):
                    if cuts[j] <= data[column][i] < cuts[j+1]:
                        new_values.append(f"{cuts[j]}-{cuts[j+1]}")
                        is_discretized = True
                        break
                        
                if not is_discretized: new_values.append(f"{cuts[j]}-{cuts[j+1]}")
                    
            my_data[column] = new_values
            
        if verbose > 0: print(f"All learned cuts: {self.cuts}")
                    
        return my_data
    
    # --------------------------------------------------------------------
    
    def discretize(self, data: pd.DataFrame, attributes: list) -> pd.DataFrame:
        
        """
        Discretizes selected attributes.
        
        Params:
            data: Information system.
            attributes: List of attributes to discretize.
            
        Return:
            DataFrame: Discretized data.
        """
        
        my_data = data.copy()
        
        for column in attributes:
        
            new_values = []
            
            for i in data.index:
                is_discretized = False
                
                for j in range(len(self.cuts[column])-1):
                    if self.cuts[column][j] <= data[column][i] < self.cuts[column][j+1]:
                        new_values.append(f"{self.cuts[column][j]}-{self.cuts[column][j+1]}")
                        is_discretized = True
                        break
                        
                if not is_discretized: new_values.append(f"{self.cuts[column][j]}-{self.cuts[column][j+1]}")
        
            my_data[column] = new_values
                    
        return my_data