"""
This is a collection of functions for calculating the distances within one domain in preparation for running a manifold alignment
It is meant to be usable all on its own, or by calling calculate distances from the ADNI Dashboard and feeding in an
ADNI_Dataset object along with a few other specifications necessary for calculating the distances
I've chosen to include the functions for using rf_gap here, although you could just as easily put them somewhere else
"""


import pandas as pd
from ADNI_Dataset_Class import ADNI_Dataset
import datetime
from mashspud import MASH
from mashspud import SPUD
import pickle
import os
import sys #for exit commands
import importlib
from tslearn.metrics import dtw_path
from dtaidistance import dtw
import numpy as np
from rfgap import RFGAP

import Progressions_Distance_Functions
importlib.reload(Progressions_Distance_Functions)


"""rf_gap distance calculating functions"""
def use_rf_proximities(tuple):
    """Creates RF proximities similarities
    
        tuple should be a tuple with position 0 being the data and position 1 being the labels"""
    
    if np.issubdtype(np.array(tuple[1]).dtype, np.integer):
        rf_class = RFGAP(prediction_type="classification", y=tuple[1], prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True)
    else:
        rf_class = RFGAP(prediction_type="regression", y=tuple[1], prox_method="rfgap", matrix_type= "dense", triangular=False, non_zero_diagonal=True)
        
    #Fit it for Data A
    rf_class.fit(tuple[0], y = tuple[1])

    #Get promities
    dataA = rf_class.get_proximities()

    # #Reset len_A and other varables
    # if self.len_A == 2:
    #     self.len_A = len(tuple[0]) 

    #     #Change known_anchors to correspond to off diagonal matricies -- We have to change this as its dependent upon A
    #     self.known_anchors_adjusted = np.vstack([self.known_anchors.T[0], self.known_anchors.T[1] + self.len_A]).T

    # elif self.len_B == 2:
    #     self.len_B = len(tuple[0])

    #Scale it and check to ensure no devision by 0
    if np.max(dataA[~np.isinf(dataA)]) != 0:

      dataA = (dataA - dataA.min()) / (dataA[~np.isinf(dataA)].max() - dataA.min()) 

    #Reset inf values
    dataA[np.isinf(dataA)] = 0

    return 1 - dataA


"""Progressions distance calculating functions"""

def euclidean_distance(sequence_1, sequence_2):
    length = len(sequence_1)
    dimensions = sequence_1.shape[1]
    #calculate the distance between the sequences with the eudlidean distance formula
    global distance
    distance = np.sqrt(np.sum((sequence_1 - sequence_2) ** 2))
    #normalize for the lenth of the sequences and the number of dimesions being compared
    global normalized_distance
    normalized_distance = distance / np.sqrt(length * dimensions)
    return normalized_distance

def path_length_dynamic_time_warping(sequence_1, sequence_2):
    # Compute DTW path and distance
    path, distance = dtw_path(sequence_1, sequence_2)
    # Normalize by path length
    normalized_distance = distance / len(path)
    return normalized_distance

def dynamic_time_warping(sequence_1, sequence_2):
    distance = dtw.distance(sequence_1.flatten(), sequence_2.flatten())
    return distance

def sequences_minimum_distance(df, id_1, id_2, method):
    """feed in the two temporal sequences that you want to compare and specify the method to 
    compare them with, eucidean for just a generalized euclidean distance, dtw for a dynamic time 
    warping, and normdtw for dtw normalized for time sequence length"""
    print(id_1) #uncomment to see if it's actually running and how fast it's going
    print(id_2)
    global person_1
    global person_2
    person_1 = df.loc[[id_1], :].to_numpy()
    person_2 = df.loc[[id_2], :].to_numpy()
    #get the possible lag values with an overlap of at least 3
    global possible_lag_values
    possible_lag_values = np.arange(-(len(person_2) - 3), (len(person_1) - 2))
    #get the set of positions for each lag and use the intersection between those sets to know how to truncate them
    global set_1
    set_1 = [x for x in range(len(person_1))]
    global distance_by_lag
    distance_by_lag = []
    for lag in possible_lag_values:
        global set_2
        set_2 = [x+lag for x in range(len(person_2))]
        global intersection
        intersection = list(set(set_1) & set(set_2)) #the positions of the overlapping values
        global person_1_truncated
        global person_2_truncated
        person_1_truncated = person_1[intersection]
        person_2_truncated = person_2[intersection - lag]
        if method == "euclidean":
            distance_by_lag.append(euclidean_distance(person_1_truncated, person_2_truncated))
        elif method == "dtw":
            distance_by_lag.append(dynamic_time_warping(person_1_truncated, person_2_truncated))
        elif method == "normdtw":
            distance_by_lag.append(path_length_dynamic_time_warping(person_1_truncated, person_2_truncated))
        else:
            print("Invalid comparison method option")
    return min(distance_by_lag)


def wrapped_euclidean_distances(df):
    """Uses the above functions to create the wrapped eucidean distances, this is one that you can call from the module"""
    # Normalize the values for each variable so that smaller units don't lend an advantage
    def normalize_column(column):
        normalized_column = column - np.mean(column)
        if np.std(column) != 0.0: #if all of the values are zero, don't be dividing by a standard deviation of zero
            normalized_column = normalized_column / np.std(column)
        return normalized_column
    global normalized_df
    normalized_df = df.apply(normalize_column, axis=0)
    # Make and return the omparisions squareform array
    rids = normalized_df.index.get_level_values("RID").unique()
    global comparisons
    comparisons = [[sequences_minimum_distance(normalized_df, id_1, id_2, method="euclidean") for id_2 in rids] for id_1 in rids]
    global squareform
    squareform = np.array(comparisons)
    # Remap the squareform distances to the range of 0 to 1
    min_val = np.min(squareform)
    max_val = np.max(squareform)
    squareform = (squareform - min_val) / (max_val - min_val)
    return squareform

def wrapped_dtw_distances(df):
    """Uses the above functions to create the wrapped dtw distances, this is one that you can call from the module"""
    # Normalize the values for each variable so that smaller units don't lend an advantage
    def normalize_column(column):
        normalized_column = (column - np.mean(column)) / np.std(column)
        return normalized_column
    normalized_df = df.apply(normalize_column, axis=0)
    # Make and return the omparisions squareform array
    rids = normalized_df.index.get_level_values("RID").unique()
    comparisons = [[sequences_minimum_distance(normalized_df, id_1, id_2, method="normdtw") for id_2 in rids] for id_1 in rids]
    squareform = np.array(comparisons)
    # Normalize to the range of 0 to 1
    min_val = np.min(squareform)
    max_val = np.max(squareform)
    squareform = (squareform - min_val) / (max_val - min_val)
    return squareform

"""General distance calculation function"""

def calculate_distances(ADNI_ds, chosen_label, distance_metric, n_pca, source_filename):
    # information to put in the viewable excel file, make a dataframe out of it
    variables = pd.Series(ADNI_ds.variables)
    rids = pd.Series(ADNI_ds.rids, dtype=int)
    vismonths = pd.Series(ADNI_ds.vismonth, dtype=int)
    labels = pd.Series(ADNI_ds.labels)
    distances_info = pd.concat([variables,rids,vismonths,labels], axis=1)
    distances_info.columns = ['Variables', 'RIDs', 'VISMONTHs', ADNI_ds.label_name]
    
    global distance_matrix
    if ADNI_ds.type == "progression":
        #it's a progressions dataset which means that we need to put it back together in order to use it
        #progressions will still do everything that a visits set will do if it isn't recognized as such
        global progressions_dataframe
        progressions_dataframe = pd.DataFrame(ADNI_ds.data)
        multi_index = pd.MultiIndex.from_arrays([rids, vismonths])
        progressions_dataframe.index = multi_index
        if distance_metric == "wrapped euclidean":
            distance_function = wrapped_euclidean_distances
        if distance_metric == "wrapped dtw":
            distance_function = wrapped_dtw_distances
        spud_object = SPUD(verbose=4, n_pca=n_pca)
        distance_matrix = spud_object.get_SGDM(data = progressions_dataframe, distance_measure = distance_function)
        ADNI_ds.store_distances(distance_matrix)
        
        #readjusts the anchors and labels for a progressions set at this point because then you can still validate a progressions set like a visits set
        global anchorlabels
        anchorlabels = pd.DataFrame({'RID': ADNI_ds.rids, ADNI_ds.label_name: ADNI_ds.labels})
        anchorlabels = anchorlabels.groupby('RID', as_index=False)[ADNI_ds.label_name].last() #if it's a visit-wise label, just take the last one
    
    elif distance_metric == "use_rf_proximities": #it's not a progression, and we're using rf_gap
        #TODO make it work for both MASH and SPUD with a few conditionals
        distance_function = use_rf_proximities
        data_tuple = (ADNI_ds.data, ADNI_ds.rf_gap_labels)
        spud_object = SPUD(verbose=4, n_pca=n_pca)
        distance_matrix = spud_object.get_SGDM(data = data_tuple, distance_measure = distance_function)
        ADNI_ds.store_distances(distance_matrix)
        
    else: #any other dataset type with any other distance measure
        #create a SPUD object and use it to calculate the distances
        spud_object = SPUD(verbose=4, n_pca=n_pca)
        distance_matrix = spud_object.get_SGDM(data = ADNI_ds.data, distance_measure = distance_metric)
        ADNI_ds.store_distances(distance_matrix)
    print(distance_matrix)
    
    #export the triangular distance matrix to a pickle file and the options selected to an excel to be retrieved later
    now = datetime.datetime.now()
    datetime_string = now.strftime("%Y-%h-%d-@-%I-%M") # we shouldn't be using this too early or too late so we don't use army time, do %H to change it back
    
    new_filename = "_".join([source_filename[:-5], chosen_label, distance_metric, datetime_string])
    export_folder = "Datasets/Distance Matricies/" + new_filename
    os.makedirs(export_folder, exist_ok=True)
    
    with open(export_folder + "/distances.pkl", 'wb') as triangular_export_file:
        pickle.dump(ADNI_ds, triangular_export_file) #dump the entire adni ds object into a binary file for retrieval
    
    # Use ExcelWriter to write the details to specific excel sheets
    with pd.ExcelWriter(export_folder + "/details.xlsx") as writer:
        distances_info.to_excel(writer, sheet_name='Details', index=False)

if __name__ == "__main__":
    distance_metric = "use_rf_proximities" #any normal metric or "wrapped euclidean", "wrapped dtw", "use_rf_proximities"
    n_pca = 10
    chosen_label = "DX_bl"
    filename = "Profile Variables 2024-12-06-19-03.xlsx"
    onehot = True
    selection = "200" #just put in the max RID that you want or 'all'
    deselected_vars = []
    
    ADNI_ds = ADNI_Dataset(filename, onehot, selection, deselected_vars, chosen_label)
    calculate_distances(ADNI_ds, chosen_label, distance_metric, n_pca, filename)
    
    feedback = f"The new distance matrix for {filename} has been created and stored in the Distance Matricies folder!\nSee the folder and associated excel file for details."
    print(feedback)