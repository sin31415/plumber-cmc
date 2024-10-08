# MLHyperTuning.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

class MLHyperTuning:

    def __init__(
        self,
        model,
        pipeline,
        cores
    ):
        """
        Initializes the model for hyper parameter tuning.

        Parameters:
        - model: The model to tune
        """
        self.model = model
        self.pipeline = pipeline
        self.cores = cores
        self.tuned_params = {}
        self.params = {}
        self.tuned_model = None

    def get_model_params(self):
        
        self.params = self.model.get_params()

        #
        print("Current model parameters:\n")
        i=1
        for param, value in self.params.items():
            print(f"{i}. {param}: {value}")
            i+=1
        
        # Create an empty dictionary to store user-input parameters for tuning

    def get_params_to_tune(self):
        params = self.params
        while True:
            # Ask user which parameter to tune
            param_to_tune = input("\nEnter the parameter you'd like to tune (or type 'done' to finish): ").strip()
    
            # If the user types 'done', exit the loop
            if param_to_tune.lower() == 'done':
                break
            
            # Check if the entered parameter is valid (exists in the model's params)
            if param_to_tune not in params:
                print(f"Invalid parameter: '{param_to_tune}'. Please enter a valid parameter.")
                continue
    
            # Ask for new values for the parameter (you can accept multiple values for GridSearchCV)
            new_value = input(f"Enter new values for '{param_to_tune}' (comma-separated if multiple): ").strip()
    
            # Convert user input to list of values if it's comma-separated, or keep as is if single value
            if ',' in new_value:
                # Convert to a list of values, remove extra spaces, and try to convert to the appropriate type
                new_value = [eval(val.strip()) for val in new_value.split(',')]
            else:
                # Convert to the appropriate type (float, int, str, etc.) for single value
                try:
                    new_value = eval(new_value)  # Evaluates to int, float, etc.
                except:
                    new_value = new_value  # Keep it as string if eval fails
    
            # Add the parameter and its new value(s) to the tuning dictionary
            self.tuned_params[param_to_tune] = new_value
        print('\nThe parameters being tested for best tuning are: \n')
        for param, value in self.tuned_params.items():
            print(f"{param}: {value}")

    def grid_tune(self):
        #Passing pipeline so we have access to processed training data
        pipeline = self.pipeline

        #Building grid
        grid = GridSearchCV(self.model, param_grid=self.tuned_params, cv=10, n_jobs=self.cores, verbose=1)

        #Fitting grid to processed pipeline data
        grid.fit(pipeline.X_train, pipeline.y_train)

        #Printing best estimator
        print(f'\nThe best estimator for {self.model} is: \n{grid.best_estimator_}')

        #Printing best score
        print(f'\nThe best score with that estimator is: \n{grid.best_score_}')

        #Getting best parameters
        best_params = grid.best_params_
        print(f'\nThe best parameters after tuning are : \n{best_params}')

        #Saving optimally tuned model
        self.tuned_model = self.model.set_params(**best_params)

    def run_tuning(self):
        """Runs the full pipeline."""
        self.get_model_params()
        self.get_params_to_tune()
        self.grid_tune()

        #Complete
        print(f'Tuning done! You can now access the object for the tuned model(.tuned_model) or retune for a different model.')