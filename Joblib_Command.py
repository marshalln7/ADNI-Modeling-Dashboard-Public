

from joblib import Parallel, delayed
import time
import importlib
import random
import os
from pushbullet import Pushbullet
random.seed(42)

#reload any of my modules that I might've changed since the last run
other_modules = ["ADNI_Dataset_Class", "Distances_Calculator", "Run_Manifold_Alignment"]
for module in other_modules:
    imported_module = importlib.import_module(module)
    importlib.reload(imported_module)

#import the actual functions that we're going to use
from ADNI_Dataset_Class import ADNI_Dataset
from Distances_Calculator import calculate_distances
from Run_Manifold_Alignment import run_manifold_alignment, run_all_alignments

"""
For the Distance Calculator:
Use any normal metric or "wrapped euclidean", "wrapped dtw", "use_rf_proximities"
"""

def send_pushbullet_notification(title, message):
    pb = Pushbullet("o.lMfHntQrCefa4z3yY0G8GjxCUE6qJWTQ")
    pb.push_note(title, message)

merged_files_list = ["Profile Variables 2025-01-03-06-35.xlsx", "Visit Variables 2025-01-03-07-12.xlsx", "Progression Variables 2024-11-09.xlsx"]



# Dictionaries of task parameters
distance_versions = [
    {"distance_metric": metric,
    "n_pca": 10,
    "chosen_label": "DX_bl",
    "rf_gap_label": "LAST_DX",
    "source_filename": "Amyloid Variables 2025-02-21-16-50.xlsx",
    "onehot": True,
    "selection": "All", #just put in the max RID that you want or 'all'
    "deselected_vars": []} for metric in ["use_rf_proximities", "euclidean"]
]

# Get all of the domains from the distance matricies folder and their unique pairings
domains_list = sorted(os.listdir("Datasets/Distance Matricies"))
domains_pairings = []
for domain1 in domains_list:
    for domain2 in domains_list:
        if domain1 != domain2 and [domain2, domain1] not in domains_pairings:
            domains_pairings.append([domain1, domain2])

manifold_combos = [
    {"model_name": "MASH",
    "domain_1_folder": pair[0],
    "domain_2_folder": pair[1],
    "n_comp": 2}
    for pair in domains_pairings 
]

manifold_combo = [
    {"model_name": "MASH",
    "domain_1_folder": "Visit Variables 2025-03-05-14-45_DIAGNOSIS_DX_use_rf_proximities_selection-ADNI3_2025-Mar-05-@-03-14",
    "domain_2_folder": "Visit Variables 2025-03-05-14-47_DIAGNOSIS_DX_use_rf_proximities_selection-ADNI3_2025-Mar-05-@-03-15",
    "n_comp": 2}
]

#results = Parallel(n_jobs=-1)(delayed(calculate_distances)(**version) for version in distance_versions)

results = Parallel(n_jobs=-1)(delayed(run_manifold_alignment)(**version) for version in manifold_combo)

print("\nAll tasks completed. Results:")
print(domains_pairings)
print(results)

send_pushbullet_notification("All Tasks Complete!", "Results:" + str(results))

