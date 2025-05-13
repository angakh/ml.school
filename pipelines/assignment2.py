from metaflow import FlowSpec, step
import mlflow


class NumericalOperationsFlow(FlowSpec):
    """
    A Metaflow pipeline that demonstrates how to track a sequence of numerical operations
    and logs results to MLflow for experiment tracking.
    """

    @step
    def start(self):
        """
        Initialize the flow with a starting number and set up MLflow tracking.
        """
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")  # Connect to local MLflow server
        mlflow.set_experiment("numerical-operations")
        
        # Start an MLflow run
        mlflow.start_run(run_name="numerical_operations_flow")
        
        # Initialize the artifact with a starting number
        self.starting_number = 10
        self.values_history = [self.starting_number]
        self.current_value = self.starting_number
        
        # Log the starting parameter
        mlflow.log_param("starting_number", self.starting_number)
        mlflow.log_metric("step_0", self.current_value)
        
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
        
        # Log the operation and result to MLflow
        mlflow.log_param("add_value", addition_value)
        mlflow.log_metric("step_1", self.current_value)
        
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
        
        # Log the operation and result to MLflow
        mlflow.log_param("multiplier", multiplier)
        mlflow.log_metric("step_2", self.current_value)
        
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
        
        # Log the operation and result to MLflow
        mlflow.log_param("subtract_value", subtraction_value)
        mlflow.log_metric("step_3", self.current_value)
        
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
        
        # Log the operation and result to MLflow
        mlflow.log_param("divisor", divisor)
        mlflow.log_metric("step_4", self.current_value)
        
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
        
        # Log final metrics to MLflow
        mlflow.log_metric("total_sum", total_sum)
        mlflow.log_metric("average", average)
        
        # Create and log a simple plot of the values history
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.values_history)), self.values_history, marker='o')
            plt.title('Numerical Operations Flow')
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.grid(True)
            
            # Add step labels
            operations = ['Initial', 'Add', 'Multiply', 'Subtract', 'Divide']
            for i, operation in enumerate(operations):
                plt.annotate(operation, (i, self.values_history[i]))
                
            # Save the plot and log it to MLflow
            plot_path = "numerical_operations_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            
        except ImportError:
            print("Matplotlib not available. Skipping plot creation.")
        
        # End the MLflow run
        mlflow.end_run()
        
        print(f"\nSummary Statistics:")
        print(f"Total Sum: {total_sum}")
        print(f"Average: {average}")
        
        print("\nFlow completed successfully!")
        print("Results have been logged to MLflow. View them at http://localhost:5000")


if __name__ == "__main__":
    NumericalOperationsFlow()