from metaflow import FlowSpec, step, card, current, Parameter
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


class Artifact:
    """A simple class to hold numeric values."""
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return str(self.value)


class ParallelFlowAssignment(FlowSpec):
    """
    A flow that demonstrates parallel processing with addition and multiplication branches.
    
    This flow initializes an artifact with a numeric value, then splits into two
    predetermined parallel branches where one adds a constant and the other 
    multiplies by a constant. Finally, a join step merges the results by 
    printing both outcomes and computing their sum.
    
    The flow uses both Metaflow and MLflow:
    - Metaflow: For workflow orchestration and @card visualizations
    - MLflow: For experiment tracking and metrics logging
    """
    
    # Define parameters that can be passed to the flow
    initial_value = Parameter('initial_value', 
                             help='Initial value for the artifact',
                             default=5)
    
    add_constant = Parameter('add_constant',
                            help='Constant to add in the addition branch',
                            default=10)
    
    multiply_constant = Parameter('multiply_constant',
                                 help='Constant to multiply in the multiplication branch',
                                 default=3)
    
    @card
    @step
    def start(self):
        """
        Initialize the flow and the artifact with a numerical value.
        
        This step:
        1. Sets up MLflow tracking
        2. Initializes the artifact with the specified value
        3. Logs the initial parameters and creates visualizations
        4. Splits the flow into two parallel branches
        """
        print(f"Starting Parallel Flow Assignment - Run ID: {current.run_id}")
        
        # Set up MLflow tracking
        experiment_name = "Parallel_Flow_Assignment"
        mlflow.set_tracking_uri("mlruns")
        
        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start an MLflow run with a unique name including the Metaflow run ID
        run_name = f"parallel_flow_{current.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        
        print(f"MLflow tracking initialized:")
        print(f"- Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"- Experiment: {experiment_name}")
        print(f"- Run ID: {self.mlflow_run.info.run_id}")
        print(f"- Run Name: {run_name}")
        
        # Log parameters to MLflow
        mlflow.log_param("initial_value", self.initial_value)
        mlflow.log_param("add_constant", self.add_constant)
        mlflow.log_param("multiply_constant", self.multiply_constant)
        mlflow.log_param("metaflow_run_id", current.run_id)
        
        # Initialize the artifact with the specified value
        print(f"Initializing artifact with value: {self.initial_value}")
        self.artifact = Artifact(self.initial_value)
        
        # Log the initial value as a metric
        mlflow.log_metric("start_value", self.initial_value)
        
        # Create a visualization of the initial value (displayed in the Metaflow card)
        plt.figure(figsize=(8, 4))
        plt.bar(["Initial Value"], [self.initial_value], color='blue')
        plt.title(f"Initial Artifact Value: {self.initial_value}")
        plt.tight_layout()
        
        # Save the plot for both Metaflow card and MLflow
        plt.savefig("initial_value_plot.png")
        mlflow.log_artifact("initial_value_plot.png")
        plt.close()
        
        # Also log our flow diagram to MLflow
        self._create_flow_diagram()
        
        # Split the flow into two parallel branches
        self.next(self.addition_branch, self.multiplication_branch)
    
    def _create_flow_diagram(self):
        """Create a visualization of the entire flow process."""
        plt.figure(figsize=(12, 8))
        
        # Initial value
        plt.text(0.5, 0.9, f"Initial Artifact\nValue = {self.initial_value}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        # Arrows to branches
        plt.arrow(0.5, 0.85, -0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
        plt.arrow(0.5, 0.85, 0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        # Branch operations
        add_result = self.initial_value + self.add_constant
        mult_result = self.initial_value * self.multiply_constant
        
        plt.text(0.25, 0.7, f"Addition Branch\n{self.initial_value} + {self.add_constant} = {add_result}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        plt.text(0.75, 0.7, f"Multiplication Branch\n{self.initial_value} × {self.multiply_constant} = {mult_result}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # Arrows to join
        plt.arrow(0.25, 0.65, 0.15, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
        plt.arrow(0.75, 0.65, -0.15, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
        
        # Join operation
        final_sum = add_result + mult_result
        plt.text(0.5, 0.5, f"Join Step\n{add_result} + {mult_result} = {final_sum}", 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='gold'))
        
        # Remove axes
        plt.axis('off')
        plt.title("Parallel Flow Visualization")
        
        # Save the diagram
        plt.savefig("flow_diagram.png")
        mlflow.log_artifact("flow_diagram.png")
        plt.close()
    
    @card
    @step
    def addition_branch(self):
        """
        First branch: Add a constant to the artifact.
        
        This step:
        1. Adds the specified constant to the artifact value
        2. Logs the operation and result to MLflow
        3. Creates visualizations for Metaflow card and MLflow
        """
        print(f"Branch 1: Adding {self.add_constant} to {self.artifact.value}")
        
        # Start a nested MLflow run for this branch
        with mlflow.start_run(run_name=f"addition_branch_{current.step_name}", nested=True):
            # Log parameters for this branch
            mlflow.log_param("operation", "addition")
            mlflow.log_param("input_value", self.artifact.value)
            mlflow.log_param("constant", self.add_constant)
            mlflow.log_param("metaflow_step", current.step_name)
            
            # Perform the addition operation
            self.result = Artifact(self.artifact.value + self.add_constant)
            
            # Log the result as a metric
            mlflow.log_metric("result", self.result.value)
            
            print(f"Branch 1 result: {self.result}")
            
            # Create visualization for the addition operation (for both Metaflow card and MLflow)
            plt.figure(figsize=(8, 4))
            plt.bar(["Input", "Constant", "Result"], 
                   [self.artifact.value, self.add_constant, self.result.value],
                   color=['blue', 'green', 'red'])
            plt.title(f"Addition Branch: {self.artifact.value} + {self.add_constant} = {self.result.value}")
            plt.tight_layout()
            
            # Save the plot
            plt.savefig("addition_branch_plot.png")
            mlflow.log_artifact("addition_branch_plot.png")
            plt.close()
        
        # Proceed to the join step
        self.next(self.join_results)
    
    @card
    @step
    def multiplication_branch(self):
        """
        Second branch: Multiply the artifact by a constant.
        
        This step:
        1. Multiplies the artifact value by the specified constant
        2. Logs the operation and result to MLflow
        3. Creates visualizations for Metaflow card and MLflow
        """
        print(f"Branch 2: Multiplying {self.artifact.value} by {self.multiply_constant}")
        
        # Start a nested MLflow run for this branch
        with mlflow.start_run(run_name=f"multiplication_branch_{current.step_name}", nested=True):
            # Log parameters for this branch
            mlflow.log_param("operation", "multiplication")
            mlflow.log_param("input_value", self.artifact.value)
            mlflow.log_param("constant", self.multiply_constant)
            mlflow.log_param("metaflow_step", current.step_name)
            
            # Perform the multiplication operation
            self.result = Artifact(self.artifact.value * self.multiply_constant)
            
            # Log the result as a metric
            mlflow.log_metric("result", self.result.value)
            
            print(f"Branch 2 result: {self.result}")
            
            # Create visualization for the multiplication operation
            plt.figure(figsize=(8, 4))
            plt.bar(["Input", "Constant", "Result"], 
                   [self.artifact.value, self.multiply_constant, self.result.value],
                   color=['blue', 'orange', 'purple'])
            plt.title(f"Multiplication Branch: {self.artifact.value} × {self.multiply_constant} = {self.result.value}")
            plt.tight_layout()
            
            # Save the plot
            plt.savefig("multiplication_branch_plot.png")
            mlflow.log_artifact("multiplication_branch_plot.png")
            plt.close()
        
        # Proceed to the join step
        self.next(self.join_results)
    
    @card
    @step
    def join_results(self, inputs):
        """
        Join step: Merge results from both branches.
        
        This step:
        1. Takes inputs from both parallel branches
        2. Computes the sum of the two branch results
        3. Logs the combined result and creates visualizations
        4. Ends the MLflow tracking
        """
        # Get results from both branches
        add_result = inputs.addition_branch.result
        multiply_result = inputs.multiplication_branch.result
        
        print("\nJoining results from both branches:")
        print(f"Addition branch result: {add_result}")
        print(f"Multiplication branch result: {multiply_result}")
        
        # Start a nested MLflow run for the join step
        with mlflow.start_run(run_name=f"join_step_{current.step_name}", nested=True):
            # Log parameters for this step
            mlflow.log_param("add_result", add_result.value)
            mlflow.log_param("multiply_result", multiply_result.value)
            mlflow.log_param("metaflow_step", current.step_name)
            
            # Compute the sum of the two branch results
            self.final_sum = add_result.value + multiply_result.value
            print(f"Sum of both branch outcomes: {self.final_sum}")
            
            # Log the final sum as a metric
            mlflow.log_metric("final_sum", self.final_sum)
            
            # Create visualization for the join step
            plt.figure(figsize=(10, 6))
            plt.bar(["Addition Result", "Multiplication Result", "Final Sum"], 
                   [add_result.value, multiply_result.value, self.final_sum],
                   color=['red', 'purple', 'gold'])
            plt.title(f"Join Step: {add_result.value} + {multiply_result.value} = {self.final_sum}")
            plt.tight_layout()
            
            # Save the plot
            plt.savefig("join_step_plot.png")
            mlflow.log_artifact("join_step_plot.png")
            plt.close()
            
            # Create comparison visualization
            self._create_comparison_visualization(add_result.value, multiply_result.value)
        
        # End the parent MLflow run
        mlflow.end_run()
        
        print(f"\nFinal result of the flow: {self.final_sum}")
        print("\nFlow completed successfully!")
        print("- View Metaflow results: Run 'metaflow ui' and navigate to this flow")
        print("- View MLflow results: Run 'mlflow ui' and navigate to the Parallel_Flow_Assignment experiment")
        
        # End the flow
        self.next(self.end)
    
    def _create_comparison_visualization(self, add_result, mult_result):
        """Create a visualization comparing the two branches side by side."""
        # Side-by-side comparison of the branches
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Addition branch visualization
        ax[0].bar(["Input", "Constant", "Result"], 
                 [self.initial_value, self.add_constant, add_result],
                 color=['blue', 'green', 'red'])
        ax[0].set_title("Addition Branch")
        ax[0].set_ylim(0, max(self.initial_value, self.add_constant, add_result) * 1.2)
        
        # Multiplication branch visualization
        ax[1].bar(["Input", "Constant", "Result"], 
                 [self.initial_value, self.multiply_constant, mult_result],
                 color=['blue', 'orange', 'purple'])
        ax[1].set_title("Multiplication Branch")
        ax[1].set_ylim(0, max(self.initial_value, self.multiply_constant, mult_result) * 1.2)
        
        plt.tight_layout()
        plt.savefig("branch_comparison.png")
        mlflow.log_artifact("branch_comparison.png")
        plt.close()
    
    @step
    def end(self):
        """
        End of the flow.
        """
        print("Flow completed successfully!")


if __name__ == "__main__":
    ParallelFlowAssignment()