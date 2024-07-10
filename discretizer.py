import pandas as pd
import numpy as np

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
        
    
    def fit(self, data: pd.DataFrame, attributes: list, number_of_output_values: int, distance_from_extreme_values=0, verbose=0) -> None:
        
        """
        Fits discretizer, i.e. generates cuts that the discretizer object will remember.
        
        Params:
            data: Information system.
            attributes: List of attributes to discretize.
            number_of_output_value: The number of values the attribute can has after discretization.
            distance_from_extreme_values: Percentage of the range by which we want to move away from the extreme data.
            verbose: Mode of generating messages by the learning process.
        
        Raise:
            ValueError: Number of output values must be greater than 1.
            ValueError: Distance from extreme values must be greater or equal 0 and lower than 0.5.
        """
        
        if 0 > distance_from_extreme_values >= 0.5:
            raise ValueError(f"Distance from extreme values must be greater or equal 0 and lower than 0.5.")
            
        if number_of_output_values <= 1:
            raise ValueError("Number of output values must be greater than 1.")
                    
        
        for column in attributes:
        
            min_value = min(data[column])
            max_value = max(data[column])
            
            data_range = max_value - min_value
            
            distance = data_range * distance_from_extreme_values
            
            if verbose > 0: print(f"{column} range: {data_range}, distance from extreme values: {distance}")
            
            cuts = []
                    
            start = min_value + distance
            end = max_value - distance
            
            if number_of_output_values > 2:
                step = (end - start) / (number_of_output_values-2)
                
                for i in range(0, number_of_output_values-1):
                    cuts.append(start + step*i)
            
            else:
                cuts.append((start+end) /2)
                
                
            self.cuts[column] = cuts
            
        if verbose > 0: print(f"All learned cuts: {self.cuts}\n")
        
            
    
    def fit_discretize(self, data: pd.DataFrame, attributes: list, number_of_output_values=5, distance_from_extreme_values=0, decimal_places=5, verbose=0) -> pd.DataFrame:
        
        """
        Fits discretizer and discretizes selected attributes.
        
        Params:
            data: Information system.
            attributes: List of attributes to discretize.
            number_of_output_value: The number of values the attribute can has after discretization.
            distance_from_extreme_values: Percentage of the range by which we want to move away from the extreme data.
            decimal_places: The number of decimal places to round the results.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            DataFrame: Discretized data.
        
        Raise:
            ValueError: Number of output values must be greater than 1.
            ValueError: Distance from extreme values must be greater or equal 0 and lower than 0.5.
        """
        
        if number_of_output_values < 1:
            raise ValueError("Number of output values must be greater than 0.")
        
        my_data = data.copy()
        
        self.fit(my_data, attributes, number_of_output_values=number_of_output_values, distance_from_extreme_values=distance_from_extreme_values, verbose=verbose)
        
        return self.discretize(my_data, decimal_places=decimal_places, verbose=verbose)
    
    # --------------------------------------------------------------------
    
    def discretize(self, data: pd.DataFrame, decimal_places=5, verbose=0) -> pd.DataFrame:
        
        """
        Discretizes selected attributes. Discretization returns string symbolizing interval. Original value was greater than or equal to the beginning of the interval and less than the end of the interval.
        
        Params:
            data: Information system.
            decimal_places: The number of decimal places to round the results.
            verbose: Mode of generating messages by the learning process.
            
        Return:
            DataFrame: Discretized data.
        
        Raise:
            RuntimeError: Discretizer must be fitted before discretize.
        """
        
        if len(self.cuts) == 0: raise RuntimeError("Discretizer must be fitted before discretize.")
        
        my_data = data.copy()
        
        attributes_to_discretize = []        
        for column in my_data.columns:
            if column in self.cuts:
                attributes_to_discretize.append(column)
                
        if verbose > 0: print(f"Attributes to discretize: {attributes_to_discretize}")
        
        for column in attributes_to_discretize:
        
            new_values = []
            
            for i in data.index:
                is_discretized = False
                
                if np.isnan(data[column][i]):
                    new_values.append(None)
                    
                if len(self.cuts[column]) > 1:    
                    for j in range(len(self.cuts[column])-1):
                        if self.cuts[column][j] <= data[column][i] < self.cuts[column][j+1]:
                            new_values.append(f"{round(self.cuts[column][j], decimal_places)}-{round(self.cuts[column][j+1], decimal_places)}")
                            is_discretized = True
                            break
                        
                if not is_discretized:
                    
                    if data[column][i] < self.cuts[column][0]:
                        new_values.append(f"<{round(self.cuts[column][0], decimal_places)}")
                    
                    if data[column][i] >= self.cuts[column][-1]:
                        new_values.append(f">={round(self.cuts[column][-1], decimal_places)}")
                  
            my_data[column] = new_values
                    
        return my_data