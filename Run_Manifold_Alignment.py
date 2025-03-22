# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:44:38 2024

@author: Marshall

This is a module to run the full manifold alignment on the precomputed distance
matricies, and to pass along the reports of everything else that's previously been
done, encapsulated in such a way that it can be utilized by both the ADNI Dashboard and
the stats servers independently
"""

import importlib
import numpy as np
import pandas as pd
import pickle
import os
import sys
import datetime

from mashspud import MASH
from mashspud import SPUD
from mashspud import triangular_helper as Triangular #used to reconstruct the triangular distance matricies that SPUD creates, see run_manifold()
from pushbullet import Pushbullet #for notifications
import traceback #to get error traceback if it fails randomly


#reload any of my modules that I might've changed
other_modules = ["ADNI_Dataset_Class"]
for module in other_modules:
    imported_module = importlib.import_module(module)
    importlib.reload(imported_module)
from ADNI_Dataset_Class import ADNI_Dataset


def generate_anchors(ADNI_domain_1, ADNI_domain_2):
    if not ADNI_domain_1.vismonth.empty and not ADNI_domain_2.vismonth.empty: #both sets have visit componants
        #lists of tuples formatted (rid, month) for each entry in each domain
        anchors1 = []
        anchors2 = []
        for rid, month in zip(ADNI_domain_1.rids, ADNI_domain_1.vismonth):
            anchors1.append((rid,month))
        for rid, month in zip(ADNI_domain_2.rids, ADNI_domain_2.vismonth):
            anchors2.append((rid,month))
    else:
        #lists of the rids for each domain
        anchors1 = list(ADNI_domain_1.rids)
        anchors2 = list(ADNI_domain_2.rids)
    
    anchors = []
    for i, rid1 in enumerate(anchors1):
        # Compare it with each item in list2
        for j, rid2 in enumerate(anchors2):
            # If they match, store the indices as a pair
            if rid1 == rid2:
                anchors.append([i, j])
                
    return anchors

def fetch_labels(ADNI_domain_1, ADNI_domain_2):
    
    if ADNI_domain_1.label_name != ADNI_domain_2.label_name:
        print("Warning, different labels used for domains A and B!")
    
    labels1 = ADNI_domain_1.labels
    labels2 = ADNI_domain_2.labels
    
    labels = pd.concat([labels1, labels2]).astype("int64") #makes it an int type to force the model not to do a regression
    return labels

def print_domain_info(ADNI_ds):
        #TODO get it to print out the specifics from the domain dataset that you selected so you 
        #know what the domain is that you're about to use to do a manifold
        #maybe even put this function in there for when you select each domain file too
        pass

def send_pushbullet_notification(api_key, title, message):
    pb = Pushbullet(api_key)
    pb.push_note(title, message)

def run_manifold_alignment(model_name, domain_1_folder, domain_2_folder, n_comp, knn = 5, heatmap=False, network=False, cross_embedding=False, verbose = False):
    #if the script fails, it'll let you know, if it succeeds, it'll let you know
    api_key = "o.lMfHntQrCefa4z3yY0G8GjxCUE6qJWTQ"
    try:
        if model_name == "SPUD":
            model = SPUD
        elif model_name == "MASH":
            model = MASH
        
        with open("Datasets/Distance Matricies/" + domain_1_folder + "/distances.pkl", 'rb') as pickle_file:
            global ADNI_domain_1
            ADNI_domain_1 = pickle.load(pickle_file)
            global square_distances_1
            square_distances_1 = Triangular.reconstruct_symmetric(ADNI_domain_1.distances)
            print_domain_info(ADNI_domain_1)
        with open("Datasets/Distance Matricies/" + domain_2_folder + "/distances.pkl", 'rb') as pickle_file:
            global ADNI_domain_2
            ADNI_domain_2 = pickle.load(pickle_file)
            global square_distances_2
            square_distances_2 = Triangular.reconstruct_symmetric(ADNI_domain_2.distances)
            print_domain_info(ADNI_domain_1)
            
        # generate the appropriate anchors
        global anchors
        global labels
        anchors = generate_anchors(ADNI_domain_1, ADNI_domain_2)
        if model_name == "MASH":
            anchors = np.array(anchors)
        if verbose == True:
            print(f"Anchors: {anchors}")
        labels = fetch_labels(ADNI_domain_1, ADNI_domain_2)
        
        #create and fit the model
        if verbose == True:
            print(f"Creating the {model_name} object...")
        if model_name == "SPUD": #gotta have some logic here because some of the parameters only apply to one or the other
            manifold_model_object = model(distance_measure_A="precomputed", distance_measure_B="precomputed", 
                                          knn=knn, OD_method = "mean", random_state = 42, verbose = (0 if not verbose else 4))
        elif model_name == "MASH":
            manifold_model_object = model(distance_measure_A="precomputed", distance_measure_B="precomputed", 
                                          knn=knn, random_state = 42, verbose = (0 if not verbose else 4))
        if verbose == True:
            print(f"Fitting the {model_name} object...")
        manifold_model_object.fit(square_distances_1, square_distances_2, known_anchors=anchors)
        if verbose == True:
            print("Fitting complete!")
        
        #get the scores and reports
        scores = manifold_model_object.get_scores(labels=labels, n_jobs=-1) #use all available cores to speed things up, n_init=1 will also make it faster but sacrifice accuracy
        scores_string = f"Fraction of Samples Closer Than the True Match: {scores[0]}\n"
        scores_string += f"Cross Embedding Score: {scores[1]}\n"
        scores_string += f"Random Forest Out of Bag Score: {scores[2]}\n"
        
        try:
            if verbose:
                print(f"{ADNI_domain_1.type} domain score: {ADNI_domain_1.distances_accuracy}")
                print(f"{ADNI_domain_2.type} domain score: {ADNI_domain_2.distances_accuracy}")
                print(scores_string)
            else:
                line1 = f"{ADNI_domain_1.type} domain score: {ADNI_domain_1.distances_accuracy}\n"
                line2 = f"{ADNI_domain_2.type} domain score: {ADNI_domain_2.distances_accuracy}\n"
                line3 = f"Random Forest Out of Bag Score: {scores[2]}\n"
                print(line1 + line2 + line3)
        except:
            pass
        

        #create the selected visualizations
        if heatmap:
            manifold_model_object.plot_heat_map()
        if network:
            manifold_model_object.plot_graphs()
        if cross_embedding:
            manifold_model_object.plot_emb(labels = labels, show_anchors = False, show_lines = False, n_componants = n_comp)
        
        #dump the resulting model object with the embedding into a pickle file
        now = datetime.datetime.now()
        datetime_string = now.strftime("%Y-%h-%d-@-%I-%M") # we shouldn't be using this too early or too late so we don't use army time, do %H to change it back

        new_filename = "_".join([model_name, domain_1_folder, domain_2_folder, datetime_string])
        
        os.makedirs("Datasets/Embeddings/" + new_filename, exist_ok=True)

        with open("Datasets/Embeddings/" + new_filename + "/class_object.pkl", 'wb') as manifold_export_file:
            pickle.dump(manifold_model_object, manifold_export_file)

        with open("Datasets/Embeddings/" + new_filename + "/domain_object_1.pkl", 'wb') as domain_1_export_file:
            pickle.dump(ADNI_domain_1, domain_1_export_file)
        
        with open("Datasets/Embeddings/" + new_filename + "/domain_object_2.pkl", 'wb') as domain_2_export_file:
            pickle.dump(ADNI_domain_2, domain_2_export_file)

        feedback = f"The {model_name} model object with the resulting embedding has been stored in the file {new_filename} in the Embeddings folder."
        if verbose:
            print(feedback)

        result = scores_string
        run_name = f"{model_name} on {domain_1_folder} and {domain_2_folder}"
        result = f"{run_name} completed with the following results:\n{result}"
        send_pushbullet_notification(api_key, "Manifold Alignment Completed", result)
    
    except Exception as e:
        send_pushbullet_notification(api_key, "Script Error", f"The following error occurred: {e}")
        traceback.print_exc()  # This will print the traceback to the console as well

    return scores[2] #returned to joblib command

def run_all_alignments():
    # create a list of domains to run against each other
    domains_list = sorted(os.listdir("Datasets/Distance Matricies"))
    domains_pairings = []
    for domain1 in domains_list:
        for domain2 in domains_list:
            if domain1 != domain2 and [domain2, domain1] not in domains_pairings:
                domains_pairings.append([domain1, domain2])

    for pair in domains_pairings:
        #runs all pairs, using the other parameters from the "if name equals main below"
        print("\n" + pair[0] + " and " + pair[1])
        run_manifold_alignment(model_name = "MASH", domain_1_folder=pair[0], domain_2_folder=pair[1], n_comp = 2)

if __name__ == "__main__": #SETTINGS WHEN RUN SEPERATELY ON THE SUPERCOMPUTER
    model_name = "MASH"
    domain1_folder = "Visit Variables 2025-03-05-14-47_DX_DX2_use_rf_proximities_selection-ADNI3_2025-Mar-06-@-09-51"
    domain2_folder = "Visit Variables 2025-03-05-16-01_DX_DX2_use_rf_proximities_selection-ADNI3_2025-Mar-06-@-09-49"
    n_comp = 2
    knn = 7
    heatmap = False
    network = False
    cross_embedding = False

    run_manifold_alignment(model_name, domain1_folder, domain2_folder, n_comp, knn, heatmap, network, cross_embedding, verbose=True)
    #run_all_alignments()

    
