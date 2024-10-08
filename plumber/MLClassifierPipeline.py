# MLClassifier.py

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
from matplotlib.gridspec import GridSpec
import seaborn as sns

class MLClassifierPipeline:
    def __init__(
        self,
        df,
        target_column,
        test_size=0.2,
        random_state=42,
        display_analytics=True,
    ):
        """
        Initializes the ML pipeline.

        Parameters:
        - df: pandas DataFrame containing the dataset.
        - target_column: The name of the target column.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        - display_analytics: Whether to display analytics for each model.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.display_analytics = display_analytics
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
        }
        self.model_results = {}
        self.best_model = None
        self.preprocessor = None
        self.trained_models = {}
        self.user_model = None

    def preprocess_data(self, drop_max_na_col=False, drop_threshold=0.25):
        """Cleans and preprocesses the data."""

        #Drops column with maximum nulls
        if drop_max_na_col==True:
            #Check for features with 25% or more NaNs and drop them
            missing_percentage = self.df.isnull().mean()
            features_to_drop = missing_percentage[missing_percentage >= drop_threshold].index
            self.df.drop(columns=features_to_drop, inplace=True)
            print(f"Dropped features with >=25% missing values: {features_to_drop.tolist()}")
            if (len(features_to_drop.tolist())/len(self.df.columns)) > 0.5:
                print(f"WARNING: More than 50% of the columns have been dropped per your threshold ({drop_threshold}).")
        
        # Drop rows with null values
        self.df.dropna(inplace=True)
        print(f"{len(self.df)} number of instances remaining.")

        # Separate features and target
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit and transform the training data, transform the test data
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_models(self):
        """Trains all models, evaluates them, and displays analytics in subplots with two matrices per row."""
    
        num_models = len(self.models)
        rows = (num_models + 1) // 2  # Calculate the number of rows needed for 2 matrices per row
    
        # Create a GridSpec layout: 2 rows per model, and 2 models per row (columns)
        fig = plt.figure(figsize=(12, rows * 6))  # Adjust height dynamically based on number of rows
        gs = GridSpec(2 * rows, 2, figure=fig)  # 2 rows per model, 2 models per row
    
        for idx, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            clf = Pipeline(steps=[('model', model)])
            clf.fit(self.X_train, self.y_train)
            y_pred_train = clf.predict(self.X_train)
            y_pred_test = clf.predict(self.X_test)
    
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            clf_report = classification_report(self.y_test, y_pred_test, output_dict=True)
    
            self.model_results[name] = {
                'model': clf,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_pred_test': y_pred_test,
            }
    
            if self.display_analytics:
                
                # Calculate the position in GridSpec
                row_idx = (idx // 2) * 2  # Two rows for each set of matrices (matrix + text)
                col_idx = idx % 2  # Switch between 0 (left column) and 1 (right column)
        
                # GridSpec setup: Confusion matrix in the top half
                ax_cm = fig.add_subplot(gs[row_idx, col_idx]);  # Top row of each pair for confusion matrix
        
                # Plot confusion matrix
                cm = confusion_matrix(self.y_test, y_pred_test)
                # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, 
                #             cbar_kws={'label': 'Scale'}, vmin=0, vmax=705, ax=ax_cm)
    
                x_axis_labels = sorted(self.df[self.target_column].value_counts().index)#["Edible", "Poisonous"]
                y_axis_labels = sorted(self.df[self.target_column].value_counts().index)#["Edible", "Poisonous"]
                sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax_cm, cmap="Purples", xticklabels=x_axis_labels, yticklabels=y_axis_labels);
                
                ax_cm.set_title(f'{name} Confusion Matrix')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_xlabel('Predicted')
        
                # Text section in the bottom half
                ax_text = fig.add_subplot(gs[row_idx + 1, col_idx])  # Bottom row of each pair for text
        
                # Add training accuracy, test accuracy, and a brief classification report in compact format
                text_info = (f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}\n"
                             f"Precision: {clf_report['weighted avg']['precision']:.4f} | "
                             f"Recall: {clf_report['weighted avg']['recall']:.4f} | "
                             f"F1-Score: {clf_report['weighted avg']['f1-score']:.4f}")
        
                ax_text.text(0.5, 0.5, text_info, ha='center', va='center', fontsize=12)
                ax_text.axis('off')  # Hide the axes for the text subplot
    
            # Store trained model
            self.trained_models[name] = clf
    
        # Adjust layout to prevent overlap
        if self.display_analytics:
            plt.tight_layout()
            plt.show()

        results_df = pd.DataFrame(self.model_results)
        display(results_df.transpose()[['train_accuracy','test_accuracy']])

    def select_best_model(self):
        """Selects the best model based on test accuracy."""
        best_accuracy = 0
        best_model_name = None

        for name, results in self.model_results.items():
            if results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model_name = name

        self.best_model = self.model_results[best_model_name]['model']
        print(f"The best model is {best_model_name} with a test accuracy of {best_accuracy:.4f}.")

    def visualize_decision_tree(self):
        """Visualizes the trained Decision Tree model."""
        # Check if the Decision Tree has been trained
        if 'Decision Tree' in self.trained_models:
            model = self.trained_models['Decision Tree'].named_steps['model']

            # Get the original class names from the target column (e.g., 'e' and 'p')
            class_names = self.y_train.unique()

            # Get the feature names from the preprocessor
            feature_names = self.preprocessor.get_feature_names_out()

            plt.figure(figsize=(40, 20))
            plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=class_names)

            if input('Press 1 to save fig') == '1':
                 plt.savefig(f'decision_tree.png', dpi=400, bbox_inches='tight')
                
            plt.title('Decision Tree Visualization')
            plt.show()

            
        else:
            print("Decision Tree model is not trained yet.")

    def get_feature_importance(self):
        """Displays feature importance for models that support it."""

        if self.best_model is None and self.user_model is None:
            print("Please run select_best_model() or user_model_return() first.")
            return
    
        if self.user_model is not None:
            model = self.user_model.named_steps['model']
        elif self.best_model is not None:
            model = self.best_model.named_steps['model']
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = pd.Series(importances, index=feature_names)
            feature_importances.sort_values(ascending=False, inplace=True)
            plt.figure(figsize=(10, 6))

            # Get a colormap and normalize
            cmap = plt.get_cmap('Purples').reversed()
            norm = plt.Normalize(vmin=0, vmax=len(feature_importances.head(20)))
            
            # Generate color for each bar
            colors = cmap(norm(np.arange(len(feature_importances.head(20)))))

            feature_importances.head(20).plot(kind='bar', color=colors)

            if input('Press 1 to save fig') == '1':
                plt.savefig(f'feature_importance.png', dpi=400, bbox_inches='tight')
            
            plt.title('Feature Importances')
            plt.show()

            
        else:
            print(f"The model {model} does not support feature importance.")
            

    def user_model_return(self):
        if self.trained_models:
            keys_list = list(self.trained_models.keys())
            print('The models trained are: \n')
            for i, name in enumerate(keys_list):
                print(f"{i}. {name}")
            choice = int(input('Which model would you like?'))

            # Access the value by index
            self.user_model = self.trained_models[keys_list[choice]]
            return self.user_model.named_steps['model']
        else:
            print('No models are trained yet. Please train the models to choose.')
    
    def run_pipeline(self, drop_max_na_col_in=False, drop_threshold_in=True):
        """Runs the full pipeline."""
        
        self.preprocess_data(drop_max_na_col=drop_max_na_col_in, drop_threshold=drop_threshold_in)
        self.train_models()
        self.select_best_model()