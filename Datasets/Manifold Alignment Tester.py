# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:29:28 2024

@author: Marshall

This code is meant to test the manifold alignment between two already created distance matricies outside of
the ADNI Dashboard
"""
import pickle
from mashspud import SPUD
from mashspud import triangular_helper as Triangular
import pandas as pd
from ADNI_Modeling_Dashboard import ADNI_Dashboard


def generate_anchors(ADNI_domain_1, ADNI_domain_2):
    global anchors1
    global anchors2
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
    
    labels = pd.concat([labels1, labels2])
    return labels


domain1_folder = "Profile Variables 2024-10-16_DX_bl_euclidean_2024-Nov-02-@-12-26"
domain2_folder = "Visit Variables 2024-10-16_DX_bl_euclidean_2024-Nov-02-@-12-27"
with open("Distance Matricies//" + domain1_folder + "//distances.pkl", 'rb') as pickle_file:
    global ADNI_domain_1
    ADNI_domain_1 = pickle.load(pickle_file)
    global square_distances_1
    square_distances_1 = Triangular.reconstruct_symmetric(ADNI_domain_1.distances)
    
with open("Distance Matricies//" + domain2_folder + "//distances.pkl", 'rb') as pickle_file:
    global ADNI_domain_2
    ADNI_domain_2 = pickle.load(pickle_file)
    global square_distances_2
    square_distances_2 = Triangular.reconstruct_symmetric(ADNI_domain_2.distances)
    
    
# generate the appropriate anchors
global anchors
global labels
anchors = generate_anchors(ADNI_domain_1, ADNI_domain_2)
labels = fetch_labels(ADNI_domain_1, ADNI_domain_2)

print("Creating the SPUD object...")
global spud_object
spud_object = SPUD(verbose=4, distance_measure_A="precomputed", distance_measure_B="precomputed", OD_method = "mean")
print("Fitting the SPUD object...")
spud_object.fit(square_distances_1, square_distances_2, known_anchors=anchors)
print("Fitting complete!")
scores = spud_object.get_scores(labels=labels)
print(f"Fraction of Samples Closer Than the True Match: {scores[0]}")
print(f"Cross Embedding Score: {scores[1]}")
print("Generating selected visualizations, see your console for output")