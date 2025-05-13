from metaflow import FlowSpec, step


class NumericalOperationsFlow(FlowSpec):
    """
    A simple Metaflow pipeline that demonstrates how to track a sequence of numerical operations.
    
    The flow starts with an initial number and applies different arithmetic operations,
    tracking the history of values along the way.
    """

    @step
    def start(self):
        """
        Initialize the flow with a starting number.
        """
        # Initialize the artifact with a starting number
        self.starting_number = 10
        self.values_history = [self.starting_number]
        self.current_value = self.starting_number
        
        print(f"Starting with initial value: {self.current_value}")
        
        # Proceed to the next step
        self.next(self.add_operation)
    
    @step
    def add_operation(self):
        """
        Add 5 to the current value.
        """
        # Perform addition
        addition_value = 5
        self.current_value += addition_value
        
        # Append the new value to the history
        self.values_history.append(self.current_value)
        
        print(f"Added {addition_value}: New value is {self.current_value}")
        
        # Proceed to the next step
        self.next(self.multiply_operation)
    
    @step
    def multiply_operation(self):
        """
        Multiply the current value by 2.
        """
        # Perform multiplication
        multiplier = 2
        self.current_value *= multiplier
        
        # Append the new value to the history
        self.values_history.append(self.current_value)
        
        print(f"Multiplied by {multiplier}: New value is {self.current_value}")
        
        # Proceed to the next step
        self.next(self.subtract_operation)
    
    @step
    def subtract_operation(self):
        """
        Subtract 7 from the current value.
        """
        # Perform subtraction
        subtraction_value = 7
        self.current_value -= subtraction_value
        
        # Append the new value to the history
        self.values_history.append(self.current_value)
        
        print(f"Subtracted {subtraction_value}: New value is {self.current_value}")
        
        # Proceed to the next step
        self.next(self.divide_operation)
    
    @step
    def divide_operation(self):
        """
        Divide the current value by 3.
        """
        # Perform division
        divisor = 3
        self.current_value /= divisor
        
        # Append the new value to the history
        self.values_history.append(self.current_value)
        
        print(f"Divided by {divisor}: New value is {self.current_value}")
        
        # Proceed to the next step
        self.next(self.end)
    
    @step
    def end(self):
        """
        Summarize all operations and calculate statistics.
        """
        # Print the entire history of values
        print("\nHistory of values:")
        for i, value in enumerate(self.values_history):
            print(f"Step {i}: {value}")
        
        # Calculate the sum and average
        total_sum = sum(self.values_history)
        average = total_sum / len(self.values_history)
        
        print(f"\nSummary Statistics:")
        print(f"Total Sum: {total_sum}")
        print(f"Average: {average}")
        
        print("\nFlow completed successfully!")


if __name__ == "__main__":
    NumericalOperationsFlow()