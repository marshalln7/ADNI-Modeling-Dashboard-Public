import pandas as pd
import numpy as np
import datetime
import re

raw_data = pd.read_csv("Datasets/Raw Data Files/UCSFFSL51 - UCSF - Longitudinal FreeSurfer (5.1) - Final Run w All A [ADNIGO,2].csv")
raw_data.sort_values(by="RID", inplace=True)

passfail_column = "OVERALLQC"
pass_indicator = "Pass"

#filter out only people who passed the quality control tests
passing_data = raw_data[(raw_data[passfail_column] == pass_indicator) | (raw_data[passfail_column].isna())]

#transform the VISCODE or VISCODE2 into VISMONTH
if "VISCODE2" in passing_data.columns:
    #replace baseline with month 0
    passing_data["VISCODE2"].replace(to_replace=["bl", "scmri", "sc"], value="m0", inplace=True)
    #create vismonth variable
    passing_data['VISMONTH'] = passing_data['VISCODE2'].str.extract('(\d+)').astype('Int64')
else: #it must be purely an ADNI1 dataset
    #replace baseline with month 0
    passing_data["VISCODE"].replace(to_replace="bl", value="m0", inplace=True)
    #create vismonth variable
    passing_data['VISMONTH'] = passing_data['VISCODE'].str.extract('(\d+)').astype('Int64')

# Reorder DataFrame
first_columns = ['RID', 'VISMONTH']
passing_data = passing_data[first_columns + [col for col in raw_data.columns if col not in first_columns]]

#keep only the columns that we need
columns_list = list(passing_data.columns)
indexers_to_keep = ["RID", "VISMONTH"]
# Regex pattern to detect which are our data columns
pattern = re.compile(r"^ST\d+[A-Z]{2}$")
# the names of the columns that we want to keep
columns_to_keep = [s for s in columns_list if s in indexers_to_keep or pattern.match(s)]
final_dataframe = passing_data[columns_to_keep]

#get rid of columns that don't have enough data in them
max_nans = 100
final_dataframe = final_dataframe.loc[:, final_dataframe.isnull().sum() <= max_nans]
final_dataframe.dropna(axis=0, how="any", inplace=True)

#export the new merged file to the Marged Data Files folder
now = datetime.datetime.now()
date_string = now.strftime("%Y-%m-%d-%H-%M")
new_filename = "Longitudinal Variables " + date_string + ".xlsx"
file_path = "Datasets/Merged Data Files/" + new_filename

final_dataframe.to_excel(file_path, index=False)