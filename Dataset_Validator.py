# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:25:44 2024

@author: Marshall
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import importlib
import os

#reload any of my modules that I might've changed
other_modules = ["ADNI_Dataset_Class"]
for module in other_modules:
    imported_module = importlib.import_module(module)
    importlib.reload(imported_module)
from ADNI_Dataset_Class import ADNI_Dataset

""" 
All functions here are made to accept a dataframe with the data columns that are to be used as
factors and a label series with all of the corresponding labels in it. It is expected that the
labels and indexers are already removed from the data dataframe
"""

def domain_validation(ADNI_dataset, model="random forest", feature_importance = False, test_set = False):
    """
    Fits the data to a random forest model and assembles all outputs into a string, that is then both
    printed and returned

    test_set specifies if you want to validate on the test set or not, keep False until you're ready to see final results
    """
    print(f"Performing {model} validation on the provided dataset...")
    
    data = ADNI_dataset.data
    feature_names = ADNI_dataset.variables
    label = ADNI_dataset.labels
    output_str = f""

    if test_set == True: # Only validate on the actual test set when this is on
        X_train, X_val, y_train, y_val = ADNI_dataset.data, ADNI_dataset.test_data, ADNI_dataset.labels, ADNI_dataset.test_labels
    else:
        X_train, X_val, y_train, y_val = ADNI_dataset.data, ADNI_dataset.val_data, ADNI_dataset.labels, ADNI_dataset.val_labels
    
    #TODO Change all of this so that it actually uses the partitions instead of cross validating because when you have multiple lines
    #for each person in a dataset it's actually cheating to do that

    if model == "random forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model == "k nearest neighbors":
        model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
        
    model.fit(X_train, y_train)
    
    # Perform validation using the validation set
    global y_pred
    y_pred = model.predict(X_val)
    validation_score = accuracy_score(y_val, y_pred)
    output_str += f'Validation score: {validation_score}\n'
    
    # Perform permutation importance analysis to identify important factors
    if feature_importance == True:
        permutation_results = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)
        for i in permutation_results.importances_mean.argsort()[::-1]:
            output_str += f"{feature_names[i]:<8} {permutation_results.importances_mean[i]:.3f} +/- {permutation_results.importances_std[i]:.3f}\n"

    # Get the sorted unique labels used in the confusion matrix FIGURE OUT WHAT TO DO WITH THIS MAYBE LABELING
    # why aren't the numbers adding up for the labels and how are there any -4 missing labels?
    unique_labels = np.unique(np.concatenate([y_train, y_val]))
    output_str += f"Represented labels: {unique_labels}\n"

    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    output_str += f'Confusion matrix: \n{conf_matrix}\n'
    
    # Print and return the constructed string
    print(output_str)
    return [validation_score, output_str]

def validate_all_datasets():
    datasets_list = sorted(os.listdir("Datasets/Merged Data Files"))
    
    for dataset in datasets_list:
        print(dataset) # Just prints the name before the output
        ADNI_ds = ADNI_Dataset(dataset, onehot, selection, deselected_vars, label_variable=chosen_label, rf_gap_variable="TOTAL13")
        domain_validation(ADNI_ds, model="random forest", feature_importance=feature_importance)
    
if __name__ == "__main__":
    chosen_label = "DIAGNOSIS"
    filename = "Progression Variables 2024-11-09.xlsx"
    onehot = True
    selection = "ADNI3" #just put in the max RID that you want or 'all'
    deselected_vars = []
    feature_importance = False

    validate_all_datasets()
    # ADNI_ds = ADNI_Dataset(filename, onehot, selection, deselected_vars, label_variable=chosen_label, rf_gap_variable="TOTAL13")
    # domain_validation(ADNI_ds, model="random forest", feature_importance=feature_importance)

    
    
    

    
