# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 09:26:58 2025

@author: Marshall
"""

import pandas as pd
import numpy as np
import sys

df = pd.read_csv("Dallan Work/visit_dx_combined.csv")

# Fill in all of the missing six month steps
patients = df['RID'].unique()
max_months = df.groupby("RID").max()["VISMONTH"]

months = []
for max in max_months:
    new_months = np.arange(0, (max + 1), 6)
    months.extend(new_months)

# Initialize an empty DataFrame to store the result
result = pd.DataFrame(columns=["RID", "VISMONTH"])

# Generate rows for each RID
for rid in patients:
    months = max_months.loc[rid]
    # Generate the range of months in 6-month intervals
    interval = list(range(0, months + 1, 6))
    # Create a DataFrame for this RID
    temp_df = pd.DataFrame({"RID": [rid] * len(interval), "VISMONTH": interval})
    # Append to the result DataFrame
    result = pd.concat([result, temp_df], ignore_index=True)

expanded_df = pd.merge(result, df, how="left", on=["RID", "VISMONTH"])

# Define variables for interpolation strategies
linear_interpolation_vars = ["TOTAL13"]

# Interpolate each group by RID
def interpolate_group(group):
    # Linear interpolation for specific variables
    group[linear_interpolation_vars] = group[linear_interpolation_vars].interpolate(method="linear")
    return group

# Apply interpolation within each RID group
df_interpolated = expanded_df.groupby("RID").apply(interpolate_group)

new_df = df_interpolated.dropna(how="any", subset=["VISITDATE"])

new_df.to_csv("Dallan Work/visit_dx_combined_updated.csv")

