# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:54:16 2024

@author: Marshall

This script creates and manages the ADNI Dashboard interface. All merging, validating, and modeling functionalities
are handled externally and imported as seen below. The ultimate goal here is to create a class that an 
optimization algorithm can then use to create the optimum model
"""


import importlib
import os
import sys
import pickle

import pandas as pd
import numpy as np
from scipy.spatial.distance import _METRICS #to choose which metric you want
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import datetime
from playsound import playsound

import Definition_Adder
import Profile_Variable_Merger
import Visit_Variable_Merger
import Progression_Variable_Merger
import Dataset_Validator
import Run_Manifold_Alignment
import ADNI_Dataset_Class
import Distances_Calculator

importlib.reload(Definition_Adder) #always looks for updates no matter what
importlib.reload(Profile_Variable_Merger)
importlib.reload(Visit_Variable_Merger)
importlib.reload(Progression_Variable_Merger)
importlib.reload(Dataset_Validator)
importlib.reload(Run_Manifold_Alignment)
importlib.reload(ADNI_Dataset_Class)
importlib.reload(Distances_Calculator)

from ADNI_Dataset_Class import ADNI_Dataset

from mashspud import MASH
from mashspud import SPUD 
#used to reconstruct the triangular distance matricies that SPUD creates, see run_manifold()
from mashspud import triangular_helper as Triangular


class ADNI_Dashboard:
    def __init__(self, master):
        self.master = master
        
        self.buttons = {}  # Dictionary to store buttons before any tabs are created
        self.dropdowns = {} # Dictionary to store the dropdowns and what has been currently selected in [dropdown, selection] format
        self.checkboxes = {}  # Dictionary to store the checkboxes and if they are selected
        self.text_entries = {} # Dictionary to store small text or number entry boxes
        self.file_selectors = {} # Dictionary to store the files that have been selected
        self.deselected_vars = [] #List for storing all of the variables that we don't want to use
        self.master.title("ADNI Modeling Dashboard - Your One Stop Shop for All Things ADNI")
        self.create_tabs() #sets up the different tabs and their corresponding pages
        
        
    def create_tabs(self):
        #creates a notebook widget in the window aka the tab headings at the top of the page
        tabs_menu = ttk.Notebook(window)
        
        #create and configure a style object for the background color
        color = "white"
        style = ttk.Style()
        style.configure("Custom.TFrame", background=color) #select your backg

        #the background image
        self.background_image = tk.PhotoImage(file="Datasets/ADNI_transparent.png")
        
        #creates multiple frame objects for the different tabs_menu
        tab1 = ttk.Frame(tabs_menu, style="Custom.TFrame")
        tab2 = ttk.Frame(tabs_menu, style="Custom.TFrame")
        tab3 = ttk.Frame(tabs_menu, style="Custom.TFrame")
        
        # put widgets into them
        self.collect_variables_frame(tab1)
        self.create_domains_frame(tab2)
        self.run_manifolds_frame(tab3)
        
        #adds those two frames as tabs_menu to the notebook widget object
        tabs_menu.add(tab1, text = "Collect Variables")
        tabs_menu.add(tab2, text = "Create Domains")
        tabs_menu.add(tab3, text = "Run Manifolds")
        #place in grid
        tabs_menu.grid(row=0, column=0)
        #select the default tab
        tabs_menu.select(tab2)
        
        # Store the tabs and tabs_menu as attributes to use later
        self.tabs_menu = tabs_menu
        self.tab1 = tab1
        self.tab2 = tab2
        self.tab3 = tab3
    
    def get_files_list(self, directory):
        #gets a list of the files in a folder, usually for selection
        try:
            return [f for f in os.listdir(directory)]
        except FileNotFoundError:
            print("Could not locate merged files...")
    
    def on_variable_select(self, event):
        # Handles updating the current selection from a listbox into a string
        selected_indices = self.variable_listbox.curselection()
        self.deselected_vars = [self.variable_listbox.get(i) for i in selected_indices]
        print("Deselected Variables:", self.deselected_vars)  # Print or store the selection
    
    def add_background(self, frame):
        #hypothetically to add background images to each of the notebook pages, but I can't get it to work and so it isn't called
        background_label = tk.Label(frame, image=self.background_image, background="white")
        background_label.place(relx=0.45, rely=0.5, anchor="center", relwidth=1, relheight=1)  # Fill the entire frame
    
    def print_header(self, frame, column_labels):
        #prints any instructions and column labels across the top in a uniform way
        for index, label in enumerate(column_labels):
            tk.Label(frame, text = label, background = "white", wraplength=200).grid(column = index, row = 0, padx = 10, pady = 10)
    
    def create_button(self, frame, name, text, column, row, command):
        """Create a button with specified name and put it in the correct frame with the right text, position, and command"""
        button = tk.Button(frame, text=text, command=command, background="white")
        self.buttons[name] = button  # Store the button in the dictionary
        button.grid(column=column, row=row, padx=10, pady=10)

    def get_button(self, name):
        #Retrieve a button by its name from the dictionary.
        return self.buttons.get(name)
    
    def create_dropdown(self, frame, name, options, default, column, row, command=None):
        selection = tk.StringVar()
        selection.set(default) # Create a tk variable to store the selection
        dropdown = tk.OptionMenu(frame, selection, command=command,  *options)
        dropdown.config(bg="white", fg="black", activebackground="lightblue") # styling options, very cool
        dropdown.grid(column=column, row=row, padx=10, pady=10) # create and place the dropdown
        self.dropdowns[name] = [dropdown, selection] # store the object and its current selection in the dictionary
    
    def get_dropdown(self, name):
        #Retrieve a dropdown by its name from the dictionary.
        return self.dropdowns.get(name)
    
    def create_checkboxes(self, frame, texts, default, column, startrow):
        for index, text in enumerate(texts): #make a checkbox for each text string recieved in the iterable texts arguement
            state = tk.BooleanVar()
            state.set(default) # Create a tk variable to store the selection
            checkbox = tk.Checkbutton(frame, text=text, variable=state, background="white")
            checkbox.grid(column=column, row=startrow+index, padx=10, pady=10)
            self.checkboxes[text] = [checkbox, state] # store the object and its current selection in the dictionary
        
    def get_checkbox(self, name):
        #Retrieve a dropdown by its name from the dictionary.
        return self.checkboxes.get(name)[1].get()
    
    def create_text_entry(self, frame, name, text, default, column, row):
        #creates a text entry box with a label saying what it is packed into the same gridspace
        #Create a frame to hold the label and entry box
        small_frame = tk.Frame(frame, background="white")
        small_frame.grid(row=row, column=column, padx=5, pady=5)

        # Label inside the frame
        label = tk.Label(small_frame, text=text, background="white", wraplength=250)
        label.pack(side='left')

        # Entry box inside the frame with default value of 10
        entry = tk.Entry(small_frame, width=8)
        entry.insert(0, default)
        entry.pack(side='left')
        self.text_entries[name] = [entry, entry.get()]

    

    def create_file_selector(self, frame, name, text, default, initial_directory, column, row, command=None, folder=False):
        """Define the file selecting behavior"""
        def select_file(initial_directory):
            # Open a file dialog with the specified initial directory
            if folder == False:
                file_path = tk.filedialog.askopenfilename(
                    initialdir=initial_directory,
                    title=name,
                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
                )
            else: #it's a directory so open it that way
                file_path = tk.filedialog.askdirectory(
                    initialdir=initial_directory,
                    title=name
                )
            # Display the selected file path
            if file_path:
                short_file_path = str(file_path).split(r"/")[-1]
                self.file_selectors[name][1] = short_file_path
                self.file_selectors[name][0].config(text=short_file_path) #update the button text to the selected file name
                #TODO make this file selector more use ambiguous so that it isn't trying to be a merge selector when it's
                #supposed to be a file selector
                if command != None: #the command being passed in must be callable and use the short file path as an argument
                    command(short_file_path)
        
        """Create a button with specified name and put it in the correct frame with the right text, position, and command"""
        button = tk.Button(frame, text=text, command=lambda: select_file(initial_directory), background="white")
        self.file_selectors[name] = [button, default]  # Store the button in the dictionary
        button.grid(column=column, row=row, padx=10, pady=10)
        
        #TODO make it so that it stores the relative file path in the file_selector dictionary correctly
        #after the file has been selected in the select_file function maybe even within the above function
        
    
    def update_variable_selection(self, new_filename):
        print(new_filename)
        #import the data from the newly selected merge file
        self.merged_file_df = pd.read_excel(r"Datasets/Merged Data Files/" + new_filename, nrows=2)
        self.variables_list = list(self.merged_file_df.columns)
        self.variables_list.remove("RID")
        # Clear the previous variable selection
        self.variable_listbox.delete(0, tk.END)
        # Add the new variable selection
        for variable_entry in self.variables_list:
            self.variable_listbox.insert(tk.END, variable_entry)
        self.print_feedback(self.feedback2, "Merge successfully selected, choose any variables that you'd like to exclude!")
            
    def print_feedback(self, text_widget, feedback):
        # Move the cursor to the end of the text in the widget
        text_widget.insert('end', feedback + '\n')
        # Scroll to the bottom to make sure the newly inserted feedback is visible
        text_widget.see('end')
    
    def add_definitions(self):
        Definition_Adder.clean_raw_files()
        self.feedback1.delete("1.0", "end")  # Delete all existing text
        self.feedback1.insert("1.0", "Labels added to the following raw data files:\n")
        #pulls the list of cleaned files from definition adder to put it in the feedback
        files_list_string = "\n".join(Definition_Adder.raw_files_list)
        self.feedback1.insert("1.0", files_list_string)
    
    def merge_profiles(self):
        Profile_Variable_Merger.merge_profile_variables()
        
    def merge_visits(self):
        Visit_Variable_Merger.merge_visit_variables()
        
    def merge_progressions(self):
        Progression_Variable_Merger.merge_progression_variables()
    
    def get_specified_dataframe(self):
        """
        This is a function for both the validation and distance matrix creation function to use to fetch the specified 
        merge with the specified filters in place. This function is meant to get all of its information from attributes
        of the ADNI_Dashboard class that record the selected options
        """
        df = 1
        
        print(df)
        return df
    
    
    def validate_domain(self):
        #df = self.get_specified_dataframe()
        filename = self.file_selectors['Select Merge'][1]
        onehot = self.checkboxes["Apply Onehot Encoding"][1].get()
        selection = self.text_entries["rid_selection"][0].get()
        chosen_label = self.dropdowns["label"][1].get()
        model = self.dropdowns["validation_method"][1].get()

        global ADNI_ds #make the dataset object global so we can look at it
        ADNI_ds = ADNI_Dataset(filename, onehot, selection, self.deselected_vars, chosen_label)
        feature_importance = self.get_checkbox("Calculate Feature Importance")
        results = Dataset_Validator.domain_validation(ADNI_ds, model=model, feature_importance=feature_importance)
        self.print_feedback(self.feedback2, results)
    
    
    def create_distance_matrix(self): 
        distance_metric = self.dropdowns["distance_metric"][1].get()
        n_pca = int(self.text_entries["n_pca"][1])
        chosen_label = self.dropdowns["label"][1].get()
        filename = self.file_selectors['Select Merge'][1]
        onehot = self.checkboxes["Apply Onehot Encoding"][1].get()
        selection = self.text_entries["rid_selection"][0].get()
        
        global ADNI_ds
        ADNI_ds = ADNI_Dataset(filename, onehot, selection, self.deselected_vars, chosen_label)
        Distances_Calculator.calculate_distances(ADNI_ds, chosen_label, distance_metric, n_pca, filename)
        
        feedback = f"The new distance matrix for {filename} has been created and stored in the Distance Matricies folder!\nSee the folder and associated excel file for details."
        self.print_feedback(self.feedback2, feedback)
    
    
    def run_manifold(self):
        # runs the manifold alignment based on the two domains selected and any other selected options
        model_name = self.dropdowns["model_selection"][1].get()
        
        # get the distance matricies to run
        domain1_folder = self.file_selectors["Domain 1"][1]
        domain2_folder = self.file_selectors["Domain 2"][1]

        n_comp = int(self.text_entries["n_comp"][1])

        heatmap = network = cross_embedding = False #all are false by default
        if self.get_checkbox("Heatmap"):
            heatmap = True
        if self.get_checkbox("Network"):
            network = True
        if self.get_checkbox("Cross Embedding"):
            cross_embedding = True

        #run the maifold alignment with the settings specified
        scores_string = Run_Manifold_Alignment.run_manifold_alignment(model_name, domain1_folder, domain2_folder, 
                                                                      n_comp, heatmap, network, cross_embedding)

        self.print_feedback(self.feedback3, scores_string)

        self.print_feedback(self.feedback3, "Manifold alignment complete!")
        
        self.print_feedback(self.feedback3, "Generating selected visualizations, see your console for output")
    
    def collect_variables_frame(self, frame):
        #adds the background image
        self.add_background(frame)
        
        #instructions for the page
        instructions = "This page merges together all of our possible variables of interest for use"
        column_labels = [instructions, "Add Labels", "Create New Merges", "Edit Variables to Merge (First Sheet Will Be Used, Remember to Hit Save!)"]
        self.print_header(frame, column_labels)
        
        #put the feedback box at the bottom of the page
        self.feedback1 = tk.Text(frame, height=15, width=80) 
        self.feedback1.grid(column = 1, row = 300, columnspan=4)
        
        #put the add labels button on the page
        self.create_button(frame, "add_labels", "Add Definitions to Raw Datasets", column = 1, row = 1, command=self.add_definitions)
        #put the merge buttons on the page
        self.create_button(frame, "merge_profile", "Merge Profile Variables", column = 2, row = 1, command=self.merge_profiles)
        self.create_button(frame, "merge_visit", "Merge Visit Variables", column = 2, row = 2, command=self.merge_visits)
        self.create_button(frame, "merge_progression", "Merge Progression Variables", column = 2, row = 3, command=self.merge_progressions)
        self.create_button(frame, "merge_profile_scan", "Merge Profile/Scan", column = 2, row = 4, command=None)
        self.create_button(frame, "merge_visit_scan", "Merge Visit/Scan ", column = 2, row = 5, command=None)
        #option to edit the variables before the merge
        self.create_button(frame, "edit_profile_variables", "Edit Profile Variables", column = 3, row = 1, command=lambda: os.startfile("Datasets\Profile Variables.xlsx"))
        self.create_button(frame, "edit_visit_variables", "Edit Visit Variables", column = 3, row = 2, command=lambda: os.startfile("Datasets\Visit Variables.xlsx"))
        self.create_button(frame, "edit_progression_variables", "Edit Progression Variables", column = 3, row = 3, command=lambda: os.startfile("Datasets\Progression Variables.xlsx"))
        
    
    def create_domains_frame(self, frame):
        #adds the background image
        self.add_background(frame)

        #print the instructions on the page
        instructions = "Select variables and distance metrics to generate domains for the manifold"
        column_labels = [instructions, "Select Merge", "Select Variables to Exclude", "Validate Domain", "Generate"]
        self.print_header(frame, column_labels)
        
        #put merge file selector on the page
        self.create_file_selector(frame=frame, name="Select Merge", text = "Select Merge", default="Profile Variables 2024-08-21.xlsx", 
                                  initial_directory="Datasets/Merged Data Files", column=1, row=1, command=self.update_variable_selection)
        
        #selecting a sample size
        self.create_text_entry(frame, "rid_selection", 'Enter the largest RID to include, the ADNI phase name, or "All":', default="All", column=1, row=2)

        #appy onehot encoding or not
        self.create_checkboxes(frame,["Apply Onehot Encoding"], default=True, column=1, startrow=3)
        
        #Variable selector listbox with natural scrolling
        self.variable_listbox = tk.Listbox(frame, selectmode="multiple")
        self.variable_listbox.grid(column=2, row=1, rowspan=3)
        # Bind selection event
        self.variable_listbox.bind("<<ListboxSelect>>", self.on_variable_select)

        #select label dropdown
        label_catalog = pd.read_csv(r"Dallan Work/visit_dx_combined.csv")
        indexers = ['RID', 'VISITDATE', 'VISMONTH']
        labels = [label for label in label_catalog.columns if label not in indexers] # will make this retrieve from options in catalog at a later date
        self.create_dropdown(frame, "label", labels, "DX_bl", column=3, row=1)

        #validation method dropdown
        validation_methods = ["random forest", "k nearest neighbors"]
        self.create_dropdown(frame, "validation_method", validation_methods, "random forest", column=3, row=2)
        
        #calculate permutation feature importance
        self.create_checkboxes(frame, ["Calculate Feature Importance"], default=False, column=3, startrow=3)
        
        #validate domains button
        self.create_button(frame, "validate_domain", "Validate Domain", column = 3, row = 4, command=self.validate_domain)
        
        #put the distance metric dropdown menu on the page
        distance_metrics = list(_METRICS.keys())
        distance_metrics.extend(["wrapped euclidean", "wrapped dtw", "use_rf_proximities"])
        self.create_dropdown(frame, "distance_metric", distance_metrics, "euclidean", column=4, row=1)
        
        #create distance matrix
        self.create_button(frame, "create_distance_matrix", "Create Distance Matrix", column = 4, row = 2, command=self.create_distance_matrix)
        
        #create the n_pca option for creating the distances
        self.create_text_entry(frame, "n_pca", "n_pca", 10, column=4, row=3)
        
        #put the feedback box at the bottom of the page
        self.feedback2 = tk.Text(frame, height=15, width=80) 
        self.feedback2.grid(column = 1, row = 300, columnspan=4) #get that sucker to the absolute bottom of the page

    
    def run_manifolds_frame(self, frame):
        #adds the background image
        self.add_background(frame)

        #print the instructions on the page
        instructions = "Select the domains and hyperparameters that you want to try to run the manifold"
        column_labels = [instructions, "Select Domains", "Select Hyperparameters", "Select Visual Outputs", "Run"]
        self.print_header(frame, column_labels)
        
        #file selectors for choosing the desired precomputed domains
        self.create_file_selector(frame=frame, name="Domain 1", text = "Domain 1", default="Profile Variables 2024-10-16_DX_bl_euclidean_2024-Oct-29-@-08-52", 
                                  initial_directory="Datasets/Distance Matricies", column=1, row=1, folder=True)
        self.create_file_selector(frame=frame, name="Domain 2", text = "Domain 2", default="Profile Variables 2024-10-16_DX_bl_euclidean_2024-Oct-29-@-08-52", 
                                  initial_directory="Datasets/Distance Matricies", column=1, row=2, folder=True)
        
        #model choice dropdown
        models = ["MASH", "SPUD"]
        self.create_dropdown(frame, "model_selection", models, "SPUD", column=2, row=1)
        
        #model options
        self.create_text_entry(frame, "n_comp", "n_comp", 2, column=2, row=2)
        
        #select outputs using checkboxes
        possible_outputs = ["Heatmap", "Network", "Cross Embedding"]
        self.create_checkboxes(frame, possible_outputs, default = False, column=3, startrow=1)
        
        #run manifold button
        self.create_button(frame, "run_manifold", "Run Manifold!", column = 4, row = 1, command=self.run_manifold)
        
        #put the feedback box at the bottom of the page
        self.feedback3 = tk.Text(frame, height=15, width=80) 
        self.feedback3.grid(column = 1, row = 300, columnspan=4) #get that sucker to the absolute bottom of the page


if __name__ == "__main__":
    window = tk.Tk()
    #window.iconbitmap('Datasets/icon_2.ico')
    app = ADNI_Dashboard(window)
    window.mainloop()

# how to get just the FOSCTTM score from the spud class
# spud_class.FOSCTTM(spud_class.block[:self.len_A, self.len_A:]

# how to get the cross embedding score
# From sklearn.manifolds import MDS
# mds = MDS(n_comp = {The amount of features you have})
# Emb = Mds.fit(spud_class.block)
# spud_class.Cross_embedding(emb, (labels1, labels2))
