import pandas as pd
import numpy as np
import datetime
import re

raw_data = pd.read_csv("Datasets/Raw Data Files/UCBERKELEY_AMY_6MM - UC Berkeley - Amyloid PET 6mm Res analysis [ADNI1,GO,2,3,4].csv")
raw_data.sort_values(by="RID", inplace=True)

#get vismonth from the baseline dates
labels_catalog = pd.read_csv("Dallan Work/visit_dx_combined_updated.csv")
visits_and_dates = labels_catalog[["RID", "VISMONTH", "VISITDATE"]]
baseline_dates = visits_and_dates[visits_and_dates["VISMONTH"] == 0][["RID", "VISITDATE"]]
baseline_dates.rename(columns={"VISITDATE": "BASELINEDATE"}, inplace=True)

#turn them into datetime columns and then merge them in for comparison
raw_data["SCANDATE"] = pd.to_datetime(raw_data["SCANDATE"])
baseline_dates["BASELINEDATE"] = pd.to_datetime(baseline_dates["BASELINEDATE"]) 
raw_data = raw_data.merge(baseline_dates, on="RID", how="left", suffixes=("", "_baseline"))
#drop entries that we don't have a baseline date for, they're all ADNI4 anyways but this could be an enhancement to come back and make
raw_data.dropna(subset=["BASELINEDATE"], inplace=True) 
raw_data['MONTH'] = (raw_data['SCANDATE'].dt.year - raw_data['BASELINEDATE'].dt.year) * 12 + (raw_data['SCANDATE'].dt.month - raw_data['BASELINEDATE'].dt.month + + (raw_data['SCANDATE'].dt.day - raw_data['BASELINEDATE'].dt.day)/30)
raw_data['VISMONTH'] = raw_data['MONTH'].apply(lambda x: round(x / 6) * 6)

# Reorder DataFrame
first_columns = ['RID', 'VISMONTH']
raw_data = raw_data[first_columns + [col for col in raw_data.columns if col not in first_columns]]

#filter out the odd tracer
raw_data = raw_data[raw_data["TRACER"] == "FBB"]

#filter out only people who passed the quality control tests
passfail_column = "qc_flag"
pass_indicator = 2
passing_data = raw_data[(raw_data[passfail_column] == pass_indicator) | (raw_data[passfail_column].isna())]

#keep only the columns that we need
columns_list = list(raw_data.columns)
indexers_to_delete = ["MONTH", "BASELINEDATE", "LONIUID", "VISCODE", "SITEID", "PTID", "SCANDATE", "PROCESSDATE", "IMAGE_RESOLUTION", "qc_flag", "TRACER", "TRACER_SUVR_WARNING", "update_stamp"]
# the names of the columns that we want to keep
columns_to_keep = [s for s in columns_list if s not in indexers_to_delete]
final_dataframe = raw_data[columns_to_keep]

#get rid of columns that don't have enough data in them
max_nans = 100
final_dataframe = final_dataframe.loc[:, final_dataframe.isnull().sum() <= max_nans]
final_dataframe.dropna(axis=0, how="any", inplace=True)

#export the new merged file to the Marged Data Files folder
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d-%H-%M")
new_filename = "Amyloid Variables " + date_string + ".xlsx"
file_path = "Datasets/Merged Data Files/" + new_filename

final_dataframe.to_excel(file_path, index=False)