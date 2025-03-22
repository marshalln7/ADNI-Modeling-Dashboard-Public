import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle
from mashspud import triangular_helper as Triangular

def validate_distance_matrix(domain_folder):
    print(domain_folder)

    with open("Datasets/Distance Matricies/" + domain_folder + "/distances.pkl", 'rb') as pickle_file:
        ADNI_domain = pickle.load(pickle_file)
                
    distance_matrix = Triangular.reconstruct_symmetric(ADNI_domain.distances)
    labels = ADNI_domain.labels

    # Create a k-NN classifier using the precomputed distance matrix
    #model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = KNeighborsClassifier(n_neighbors=7, metric="precomputed")

    # Perform cross-validation to evaluate performance
    scores = cross_val_score(model, distance_matrix, labels, cv=3)

    # Print accuracy scores
    # print(" Cross-validation accuracy scores:", scores)
    print(" Mean accuracy:", scores.mean())

if __name__ == "__main__":
    # Validates all of the distance matricies in the distance matricies folder
    domains_list = sorted(os.listdir("Datasets/Distance Matricies"))

    for folder in domains_list:
        validate_distance_matrix(domain_folder=folder)