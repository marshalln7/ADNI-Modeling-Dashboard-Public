"""
This is a collection of functions for calculating the distances within one domain in preparation for running a manifold alignment
It is meant to be usable all on its own, or by calling calculate distances from the ADNI Dashboard and feeding in an
ADNI_Dataset object along with a few other specifications necessary for calculating the distances
I've chosen to include the functions for using rf_gap here, although you could just as easily put them somewhere else
"""

import importlib
import pandas as pd
import datetime
from mashspud import MASH
from mashspud import SPUD
import pickle
import os
import sys #for exit commands
from tslearn.metrics import dtw_path
from dtaidistance import dtw
import numpy as np
from pushbullet import Pushbullet #for notifications
import traceback #to get error traceback if it fails randomly
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from mashspud import triangular_helper as Triangular

#reload any of my modules that I might've changed
other_modules = ["Progressions_Distance_Functions", "ADNI_Dataset_Class", "rfgap", "Dataset_Validator"]
for module in other_modules:
    imported_module = importlib.import_module(module)
    importlib.reload(imported_module)
from ADNI_Dataset_Class import ADNI_Dataset
from rfgap import RFGAP
from Dataset_Validator import domain_validation

"""rf_gap distance calculating function to pass into MASH or SPUD"""

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


"""Progressions distance calculating functions to pass into MASH or SPUD"""

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

def calculate_distances(source_filename, onehot, selection, deselected_vars, chosen_label, distance_metric, n_pca, rf_gap_label):
    #if the script fails, it'll let you know, if it succeeds, it'll let you know
    api_key = "o.lMfHntQrCefa4z3yY0G8GjxCUE6qJWTQ"
    try:
        #first, put make a ADNI Dataset object out of the provided selection
        ADNI_ds = ADNI_Dataset(source_filename, onehot, selection, deselected_vars, chosen_label, rf_gap_variable=rf_gap_label)
        
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

            #store the original dataframe for use later
            ADNI_ds.progressions_dataframe = progressions_dataframe

            # Group by RID and re-index each group to the maximum size
            grouped = progressions_dataframe.groupby('RID')
            max_size = grouped.size().max()

            # Apply reindex to ensure uniform size for all groups
            reindexed = grouped.apply(
                lambda x: x.reset_index(drop=True).reindex(range(max_size))
            )

            # Reshape to wide format
            reshaped = reindexed.unstack()

            # Convert MultiIndex column tuples into just string column labels
            reshaped.columns = ['{}_{}'.format(col, i) for col, i in reshaped.columns]

            ADNI_ds.data = np.array(reshaped)
            
            #readjusts the anchors and labels for a progressions set at this point because then you can still validate a progressions set like a visits set
            #we're assuming that the test data from a progressions set isn't going to be useful later, so it isn't reconstructed but you could do it in the same way
            anchorlabels = pd.DataFrame({'RID': ADNI_ds.rids, ADNI_ds.label_name: ADNI_ds.labels})
            anchorlabels = anchorlabels.groupby('RID', as_index=False)[ADNI_ds.label_name].last() #if it's a visit-wise label, just take the last one
            ADNI_ds.rids = anchorlabels.RID
            ADNI_ds.labels = anchorlabels[ADNI_ds.label_name]
        
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

        # VALIDATE THE DATASET
        if ADNI_ds.type != "progression":
            ADNI_ds.dataset_accuracy = domain_validation(ADNI_ds, model="random forest", feature_importance=False, test_set=True)[0]
            print("Don't forget that this is validating for the whole dataset, for publicaion use come back and try to subset")
        else:
            validate_ADNI_ds = ADNI_Dataset(source_filename, onehot, selection, deselected_vars, label_variable=chosen_label, rf_gap_variable="TOTAL13")
            ADNI_ds.dataset_accuracy = domain_validation(validate_ADNI_ds, model="random forest", feature_importance=False, test_set=True)[0]

        # VALIDATE THE DISTANCE MATRIX
        distance_matrix = Triangular.reconstruct_symmetric(ADNI_ds.distances)
        labels = ADNI_ds.labels
        # Create a k-NN classifier using the precomputed distance matrix
        knn = KNeighborsClassifier(n_neighbors=7, metric="precomputed")
        # Perform cross-validation to evaluate performance (we'll need to fix this later to keep whole people in train or test)
        scores = cross_val_score(knn, distance_matrix, labels, cv=3)
        # Print and save accuracy scores
        print(" Distances cross-validation scores:", scores)
        print(" Distances mean accuracy:", scores.mean())
        ADNI_ds.distances_accuracy = scores.mean()
        
        #export the triangular distance matrix to a pickle file and the options selected to an excel to be retrieved later
        now = datetime.datetime.now()
        datetime_string = now.strftime("%Y-%h-%d-@-%I-%M") # we shouldn't be using this too early or too late so we don't use army time, do %H to change it back
        selection_string = f"selection-{selection}"

        new_filename = "_".join([source_filename[:-5], chosen_label, rf_gap_label, distance_metric, selection_string, datetime_string])
        export_folder = "Datasets/Distance Matricies/" + new_filename
        os.makedirs(export_folder, exist_ok=True)
        
        with open(export_folder + "/distances.pkl", 'wb') as triangular_export_file:
            pickle.dump(ADNI_ds, triangular_export_file) #dump the entire adni ds object into a binary file for retrieval
        
        # Use ExcelWriter to write the details to specific excel sheets
        with pd.ExcelWriter(export_folder + "/details.xlsx") as writer:
            distances_info.to_excel(writer, sheet_name='Details', index=False)

        feedback = f"The new distance matrix for {source_filename} with size {ADNI_ds.distances.shape} has been created and stored in the Distance Matricies folder! See the folder and associated excel file for details."
        print(feedback)
        send_pushbullet_notification(api_key, "Distances Calculation Completed", feedback)
        
    except Exception as e:
        send_pushbullet_notification(api_key, "Script Error", f"The following error occurred: {e}")
        traceback.print_exc()  # This will print the traceback to the console
    

def send_pushbullet_notification(api_key, title, message):
    pb = Pushbullet(api_key)
    pb.push_note(title, message)

if __name__ == "__main__":
    distance_metric = "use_rf_proximities" #any normal metric or "wrapped euclidean", "wrapped dtw", "use_rf_proximities"
    n_pca = 10
    chosen_label = "DX"
    rf_gap_label = "DX2"
    filename = "Visit Variables 2025-03-05-14-47.xlsx"
    onehot = False
    selection = "ADNI3" #just put in the max RID that you want or 'all'
    deselected_vars = []

    calculate_distances(filename, onehot, selection, deselected_vars, chosen_label, distance_metric, n_pca, rf_gap_label=rf_gap_label)