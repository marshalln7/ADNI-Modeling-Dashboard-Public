# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:44:38 2024

@author: Marshall

This is a module to run the full manifold alignment on the precomputed distance
matricies, and to pass along the reports of everything else that's previously been
done, encapsulated in such a way that it can be utilized by both the ADNI Dashboard and
the stats servers independently
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
from ADNI_Dataset_Class import ADNI_Dataset
from mashspud import MASH
from mashspud import SPUD
#used to reconstruct the triangular distance matricies that SPUD creates, see run_manifold()
from mashspud import triangular_helper as Triangular

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
                
    print(anchors)
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

def run_manifold_alignment(model_name, domain_1_folder, domain_2_folder, n_comp, 
                           heatmap=False, network=False, cross_embedding=False):
    
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
    labels = fetch_labels(ADNI_domain_1, ADNI_domain_2)
    
    #create and fit the model
    print(f"Creating the {model_name} object...")
    global manifold_model_object
    if model_name == "SPUD": #gotta have some logic here because some of the parameters only apply to one or the other
        manifold_model_object = model(verbose=4, distance_measure_A="precomputed", distance_measure_B="precomputed", OD_method = "mean")
    elif model_name == "MASH":
        manifold_model_object = model(verbose=4, distance_measure_A="precomputed", distance_measure_B="precomputed")
    print(f"Fitting the {model_name} object...")
    manifold_model_object.fit(square_distances_1, square_distances_2, known_anchors=anchors)
    print("Fitting complete!")
    
    #get the scores and reports
        
    scores = manifold_model_object.get_scores(labels=labels, n_comp=n_comp)
    scores_string = f"Fraction of Samples Closer Than the True Match: {scores[0]}\n"
    scores_string += f"Cross Embedding Score: {scores[1]}\n"
    scores_string += f"Random Forest Out of Bag Score: {scores[2]}\n"
    print(scores_string)
    
    #create the selected visualizations
    if heatmap:
        manifold_model_object.plot_heat_map()
    if network:
        manifold_model_object.plot_graphs()
    if cross_embedding:
        manifold_model_object.plot_emb(labels = labels, show_anchors = False, show_lines = False, n_comp = 2)
    
    return scores_string

if __name__ == "__main__": #SETTINGS WHEN RUN ON THE SUPERCOMPUTER
    
    model_name = "MASH"
    domain1_folder = "Profile Variables 2024-11-25-12-43_DX_bl_euclidean_2024-Dec-02-@-01-31"
    domain2_folder = "Profile Variables 2024-11-25-12-43_DX_bl_euclidean_2024-Dec-02-@-01-31"
    n_comp = 2
    heatmap = False
    network = False
    cross_embedding = False
    
    run_manifold_alignment(model_name, domain1_folder, domain2_folder, 
                           n_comp, heatmap, network, cross_embedding)
