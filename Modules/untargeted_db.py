# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:41:24 2024

@author: Ronan
"""
import os
import pickle
import lz4.frame
import pandas as pd
from pyteomics import mgf
from datetime import datetime


class UntargetedDB():
    def __init__(self, DBname_path):
        self.name = os.path.basename(DBname_path)
        self.path = DBname_path
        self.ion_cluster = pd.DataFrame({'ic_id', 'specie', 'feature', 'setting', 'experiment', 'treatment'}) # contain ion_clusters with corresponding faetures in lists. 
        self.feature = pd.DataFrame({'feature_id', 'pepmass', 'rtinseconds', 'mslevel', 'charge', 'neutral', 'isotopes'}) # contain features from the ion clusters with MGF associated data. Be careful, a same m/z can correspond to diferent features coming from different experiments
        self.specie = ['Allobates_femoralis',
                        'Brachypodium_distachyon',
                        'Caenorhabditis_elegans',
                        'Diabrotica_balteata',
                        'Diabrotica_virgifera',
                        'Homo_sapiens',
                        'Mus_musculus',
                        'Mangifera_indica',
                        'Oryza_sativa',
                        'Zea_mays',
                        'Soil']
        self.setting = ['untargeted + targeted normal + pos',
                         'untargeted + targeted normal + neg',
                         'untargeted + targeted reverse + pos',
                         'untargeted + targeted reverse + neg',
                         'targeted normal + pos',
                         'targeted normal + neg',   
                         'targeted reverse + pos',  
                         'targeted reverse + neg',  
                         'untargeted normal + pos',
                         'untargeted normal + neg',
                         'untargeted reverse + pos',
                         'untargeted reverse + neg'] # type of lcms settings
        self.experiment = pd.DataFrame({'exp_id', 'analysis_date', 'folder_path', 'template_path'}) # if no template path because < 2023, template will be None. analysis_date correspond to the date of data analysis, because same data can be analyzed multiple times
        
    
    # Function to find neutral masses of features in the new ion clusters 
    def find_neutral(self, Featurelistpath):
        
        pass
    
    # Function to update the ion clusters adn find potential neutral masses and precusor ion in the ion clusters
    def update_neutral(self):
        pass
    
    # Function to open an untargeted DB
    def open_db(self):
        with lz4.frame.open(self.path, "rb") as file:
            loaded_db = pickle.load(file)    
        
        
    # Function to add new data in the db
    def add_db(self, Featurelistpath, Ic_table, MGF_path, Sample_template_path, Setting, Specie, Stat_result): # Ion_cluster_list is the new ion_cluster from the current analysis
        with open(MGF_path, 'r') as file:
            mgf = file.readlines()        
        current_date = datetime.now()            
        formatted_date = current_date.strftime("%d_%m_%Y") # Format the date as DD_MM_YYYY
        if Sample_template_path:            
            if Sample_template_path not in self.experiment['template_path']:
                new_exp = {'id': len(self.experiment), 'analysis_date': formatted_date, 'folder_path': Featurepath, 'template_path': Sample_template_path}
                self.experiment.append(new_exp, ignore_index=True)
        else:
            new_exp = {'id': len(self.experiment), 'analysis_date': formatted_date, 'folder_path': Featurepath, 'template_path': 'No'}
            self.experiment.append(new_exp, ignore_index=True)
        
        
        for index, ic in Ic_table.iterrows():
            sd_table = ic['sd_table']
            
            
            
        new_feature = {'feature_id': len(self.feature), 'pepmass', 'rtinseconds', 'mslevel', 'charge', 'neutral', 'isotopes'}
        self.feature.append(new_feature, ignore_index=True)
                            
                
                 
                
        
        
    # Function to make a db search    
    def match_search(self):
        
    
    
    