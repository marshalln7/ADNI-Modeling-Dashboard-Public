"""This is a super handy object for storing absolutely everything about the dataset in question so that it can be passed and referenced easily"""

import pandas as pd
import numpy as np
import sys

class ADNI_Dataset:
    """
    A class that doesn't check anything from the dashboard but in fed a dataframe and label 
    name and creates the object from there
    """
    def __init__(self, filename, onehot, selection, deselected_vars, label_variable = "none", rf_gap_variable = "TOTAL13"):
        print("Creating ADNI Dataset object...")
        global dataframe
        dataframe = pd.read_excel(r"Datasets/Merged Data Files/" + filename)
        dataframe['RID'] = dataframe['RID'].fillna(method='ffill') #fills back in the multilevel index so each row has an RID, if applicable
        self.type = filename.split()[0].lower()
        
        #filter out just the selection specified
        if selection.startswith("ADNI"): # the user is specifying an ADNI phase
            dataframe = dataframe[dataframe["PHASE"] == selection]
        elif selection.isdigit(): # they specified a maximum RID
            dataframe = dataframe.loc[dataframe['RID'] <= int(selection)]
        elif selection.lower() == "all": # they want everything
            pass
        else:
            print("No valid selection of RIDs was made, and so all will be included")
        
        if filename.startswith("Profile") == True: #there are multiple entries for each person here if they participated in multiple ADNI phases
            dataframe = dataframe.groupby('RID').first().reset_index()
            
        
        if filename.startswith("Scans") == False: #variable selection and modification doesn't apply to scans
            #exclude deselected variables
            dataframe.drop(deselected_vars, axis=1, inplace=True)
            
            #load variable types catalog so we know what to onehot encode
            variable_types = pd.read_excel(r"Datasets/Merged Data Files/" + filename, sheet_name="Variable Catalog")
            #remove deselected varaibles from the catalog so we don't try to onehot encode them
            variable_types = variable_types.loc[~variable_types['Variable'].isin(deselected_vars)]
            #does the onehot encoding
            if onehot == True:
                categorical_variables_list = list(variable_types.loc[variable_types["Type"] == "Categorical"].Variable)
                dataframe = pd.get_dummies(dataframe, dummy_na=True, columns=categorical_variables_list, dtype=float)
        
        raw_label_catalog = pd.read_csv(r"Dallan Work/visit_dx_combined_updated.csv")

        #only leaves the label variable and indexers for merging in
        if "VISMONTH" in dataframe.columns:
            label_catalog = raw_label_catalog[["RID", "VISMONTH", label_variable]]
            rf_label_catalog = raw_label_catalog[["RID", "VISMONTH", rf_gap_variable]]
        else:
            label_catalog = raw_label_catalog[["RID", label_variable]]
            rf_label_catalog = raw_label_catalog[["RID", rf_gap_variable]]

        #TODO Have it do the exact same thing that it does adding and then storing the label variable with the rf_gap variable for use when you do use_rf_proximities
        
        #adds the labels by left merging the data with the labels catalog
        global dataframe_with_labels
        for variable, catalog in zip([label_variable, rf_gap_variable], [label_catalog, rf_label_catalog]):
            if variable in dataframe.columns: #determines if the label is just in the dataset
                dataframe_with_labels = dataframe #no need to add anything
            elif "VISMONTH" in dataframe.columns: #we're pairing labels to month-wise measurements
                label_catalog_by_visit = catalog.groupby(['RID',"VISMONTH"]).first().reset_index()
                dataframe_with_labels = pd.merge(dataframe, label_catalog_by_visit, how='left', on=['RID', 'VISMONTH'])
            else: #we're pairing labels to person-wise measurements
                label_catalog_by_rid = catalog.groupby('RID').first().reset_index()
                dataframe_with_labels = pd.merge(dataframe, label_catalog_by_rid, how='left', on=['RID'])
            #must fill in for nans because there could now fake vismonths in the set that won't have a matching label
            dataframe_with_labels[variable] = dataframe_with_labels[variable].fillna(method="ffill") 
        
        
        self.label_name = label_variable #store the label variable name for later
        self.rf_gap_variable_name = rf_gap_variable #store the rf_gap variable name for later
        
        #Delete entries without a label to them
        dataframe_with_labels = dataframe_with_labels.dropna(subset=[label_variable])

        #This is the part where we PARTITION up our data but still store it in the object
        
        #Get the RIDs associated with each group
        split = pd.read_csv("Datasets/data_split.csv")
        split['RID'] = split['ID'].apply(lambda x: float(x.split("_")[-1]))
        split.sort_values(by = ["RID"], inplace = True)
        train_list = list(split[split["Split"] == "Train"]["RID"])
        validation_list = list(split[split["Split"] == "Validation"]["RID"])
        test_list = list(split[split["Split"] == "Test"]["RID"])
        
        #Stores the validation data and labels seperately for later use
        train_dataframe_with_labels = dataframe_with_labels[dataframe_with_labels["RID"].isin(train_list)]
        val_dataframe_with_labels = dataframe_with_labels[dataframe_with_labels["RID"].isin(validation_list)] #we're done with these two so we're just gonna turn them into numpy arrays now
        test_dataframe_with_labels = dataframe_with_labels[dataframe_with_labels["RID"].isin(test_list)]
        
        #seperates the labels from each of the partitions and stores them seperately
        self.labels = train_dataframe_with_labels.pop(label_variable)
        self.val_labels = val_dataframe_with_labels.pop(label_variable)
        self.test_labels = test_dataframe_with_labels.pop(label_variable)
        self.rf_gap_labels = train_dataframe_with_labels.pop(rf_gap_variable)
        self.val_rf_gap_labels = val_dataframe_with_labels.pop(rf_gap_variable)
        self.test_rf_gap_labels = test_dataframe_with_labels.pop(rf_gap_variable)
        
        dataframe = train_dataframe_with_labels #it doesn't have labels anymore so store it back as 'dataframe'
        
        #try to store the rids and vismonths of the test data and generate appropriate anchors
        try:
            self.rids = dataframe.pop("RID")
        except KeyError:
            print("Error, did you try to feed the ADNI Dataset object an indexed dataframe? Because I can't find any RIDs")
        try: #tries to pull out the VISMONTH column, returns an empty series if there isn't one
            self.vismonth = dataframe.pop("VISMONTH")
        except:
            self.vismonth = pd.Series(dtype=float) #an empty series
        try: #let this one be a try because I'm not sure if I actually want to work with ADNI phases or not as opposed to just RID maximums
            self.phases = dataframe.pop("PHASE")
        except:
            self.phases = None

        #removes anything from the variable catalog that still remains (everything but the unused labels should already be gone)
        global catalog_columns
        catalog_columns = list(raw_label_catalog.columns)
        catalog_columns.append("PHASE")
        dataframe.drop(columns=catalog_columns, inplace=True, errors="ignore")
        #does the same with the validation and test data, they still have indexers that will be deleted but we don't need to store them
        val_dataframe = val_dataframe_with_labels.drop(columns=catalog_columns, errors="ignore")
        test_dataframe = test_dataframe_with_labels.drop(columns=catalog_columns, errors="ignore")
        
        #dataframe = self.drop_object_columns(dataframe) We shouldn't need this
        self.variables = list(dataframe.columns)
        
        self.data = np.array(dataframe)
        self.val_data = np.array(val_dataframe)
        self.test_data = np.array(test_dataframe)

        #playsound('finished_harp.mp3')
    
    def drop_object_columns(self, df):
        # Select columns with dtype 'object'
        object_cols = df.select_dtypes(include=['object']).columns
        # Print message showing deleted columns
        if not object_cols.empty:
            print(f"Deleted object-type columns: {', '.join(object_cols)}")
        else:
            print("No object-type columns found to delete.")
        # Drop the object-type columns
        df = df.drop(columns=object_cols)
        return df
    
    def store_distances(self, distances):
        """Only stores the triangular distances by default, this is for if you wanna change that later"""
        self.distances = distances
        

if __name__ == "__main__":
    chosen_label = "DX_bl"
    filename = "Profile Variables 2024-12-06-19-03.xlsx"
    onehot = False
    selection = "400" #just put in the max RID that you want
    deselected_vars = []
    
    ADNI_ds = ADNI_Dataset(filename, onehot, selection, deselected_vars, chosen_label)
    rids = ADNI_ds.rids
    labels = ADNI_ds.labels