# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:42:55 2023

@author: Ronan
"""

import os
import re
import pickle
import lz4.frame
import pandas as pd
import matplotlib.pyplot as plt
import Modules.cache_manager as cache
import Modules.CLI_text as cli

MassLearn_directory = r'D:\Metabolomic\MassLearn 2.2'
#TODO : write here the masslearn folder with the installation software tool

class FileManager():
    """
    Class to manage saving NEW file, from different extensions
    Use as the following example:
        object = FileManager('batch_1', '../Batch_files', xmltreeparser, 'xml')
        object.saving_file(Log) # will save the file with desired name in desired location in desired format extension
    """
    def __init__(self, Filename = '', Folder = '', Data = '', Extension = ''): # Filename is path/name.extension, Folder is absolute or relative (from MassLearn/) folder where the file is saved, Data is the raw data to be written, Extension can be 'csv', '0ds', 'txt' and 'xml'
        self.data = Data # Data of the file
        self.filename = Filename
        self.folder = os.path.abspath(Folder)
        if Extension not in ['_ms1', '_ms2']:
            self.extension = '.' + Extension
            self.path = os.path.join(self.folder, self.filename + self.extension)
        else:
            self.extension = f'{Extension}.csv'
            self.path = os.path.join(self.folder, self.filename + self.extension)
    
    # Function to manage file overwriting and copy
    def handle_existing_file(self):
        """
        Handles existing files with a prompt to overwrite or rename.
        If renamed, adds '- copy', '- copy (2)', '- copy (3)', etc. to the filename.
        """
        base, extension = os.path.splitext(self.path)
        i = 0
        while True: # while loop to check if there is already one or multiple presence of "filename - copy(nb).extension"
            if i == 0:
                new_path = self.path
            else:
                new_path =  f"{base} - copy ({i}){extension}"
            if os.path.isfile(new_path):
                i += 1
            else:
                new_path = os.path.splitext(os.path.basename(new_path))[0] # the base filename
                break
        answer = self.input_error(['o', 'r', 'c'], f'"{self.filename}" already exists. Overwrite (o) or rename (r)? ')
        if answer.lower() == 'o':
            return self.filename
        elif answer.lower() == 'r':
            answer = self.input_error(['y', 'n', 'c'], f'Do you want to rename as "{new_path}" (y), enter a new name (n)?')
            if answer.lower() == 'y': 
                return new_path
            elif answer.lower() == 'n':
                while True:
                    new_file = input('\nEnter a new file name (without extension): ')
                    if self.is_valid_filename(new_file) == True and new_file != '':
                        break
                    else:
                        print(f"\nThe following characters are not allowed in filenames: {self.is_valid_filename(new_file)}\n")
                return new_file
            
    # Function to manage file overwriting and copy
    def handle_existing_file_dash(self):
        if os.path.isfile(self.path):
            return False
        else:
            return True
            
        
    # Function to manage saving new files
    def saving_file(self, Log, Handle = True): 
        new_path = self.path
        if Handle == True: # Handle is a safeguard for the files which need to be tested. Is the name have already been validated with validity()y, it is useless (typically for feature list and batch files generation)
            if os.path.isfile(self.path):
                new_path = self.handle_existing_file()
                if new_path is None:
                    log = "Saving canceled."
                    Log.update(log)
                    print(f"\n{log}")
                    return None # TODO - close the current window and go back to precedent menu
                else: # saving file with new path
                    new_path = os.path.join(self.folder, new_path + self.extension) 
                    base = os.path.splitext(new_path)[0] # take only the name, not the extension
                    self.writing_protocol(base, Log)
            else:
                base = os.path.splitext(self.path)[0] # take only the name, not the extension
                self.writing_protocol(base, Log)
        else:
            base = os.path.splitext(self.path)[0] # take only the name, not the extension
            self.writing_protocol(base, Log)               
        return new_path
    
    # Function to manage saving new files
    def saving_file_dash(self, Log, Project = None): 
        base = os.path.splitext(self.path)[0] # take only the name, not the extension
        Log = self.writing_protocol(base, Log, Project)               
        return Log
    
    # Function to use the good protocol when writing a new file
    def writing_protocol(self, Path, Log, Project = None):
        if self.extension == '.csv':
            self.data.to_csv(f'{Path}.csv', index = False)
            log = f'{os.path.basename(Path)}.csv saved.'
            Log.update(log, Project)
        elif self.extension == '_ms1.csv':
            suffix = Path.split('_')[-1]
            if suffix != 'ms1.csv':
                self.data.to_csv(f'{Path}_ms1.csv', index = False)
                log = f'{os.path.basename(Path)}_ms1.csv saved.'
            else:
                self.data.to_csv(f'{Path}.csv', index = False)
                log = f'{os.path.basename(Path)}.csv saved.'
            Log.update(log, Project)
        elif self.extension == '_ms2.csv':
            suffix = Path.split('_')[-1]
            if suffix != 'ms2.csv':
                self.data.to_csv(f'{Path}_ms2.csv', index = False)
                log = f'{os.path.basename(Path)}_ms2.csv saved.'
            else:
                self.data.to_csv(f'{Path}.csv', index = False)
                log = f'{os.path.basename(Path)}.csv saved.'
            Log.update(log, Project)
        elif self.extension == '.xml':
            self.data.write(f'{Path}.xml', pretty_print=True, encoding='UTF-8')   
            log = f'{os.path.basename(Path)}.xml saved.'
            Log.update(log, Project)
        elif self.extension == '.html':
            pass # html files are usually saved already in another object
            
        elif self.extension == '.ods':
            self.data.save(f'{Path}.ods')
            log = f'{os.path.basename(Path)}.ods saved.'
            Log.update(log, Project)
        elif self.extension == '.txt':
            with open(f'{Path}.txt', "w") as file:
                file.write(self.data)
            log = f'{os.path.basename(Path)}.txt saved.'
            Log.update(log, Project)
        elif self.extension == '.masslearn':   
            with lz4.frame.open(f'{Path}.masslearn', "wb") as file:
                try:
                    pickle.dump(self.data, file)
                except pickle.PicklingError as error:
                    text = f'An error occurred while saving the project: {error}'
                    text = cli.DisplayText(text)
                    print(text.simple())
            log = f'{os.path.basename(Path)}.masslearn saved.'
            Log.update(log, Project)
        return Log
            
    
    # Function for input error, only for strings
    def input_error(self, Valid, Question): # Valid is the list of valid inputs
        while True:
            answer = input(f'\n{Question}\n\n---> ')
            if answer.lower() in Valid: # if user wants to quit
                break
        return answer
    
    # Function to check validity of file names
    def is_valid_filename(self, Name):
        """
        Returns True if the given filename is a valid filename in Windows or Linux.
        Returns False otherwise.
        """
        # Define the forbidden characters and reserved words
        forbidden_chars = r'[<>:"/\\|?*\x00-\x1f.]'
        reserved_words = r'(CON|PRN|AUX|NUL|COM\d|LPT\d)(\..*)?'
    
        # Check if the filename matches the forbidden characters or reserved words
        if re.search(forbidden_chars, Name) or re.match(reserved_words, Name, re.IGNORECASE):
            return ''.join(set(re.findall(forbidden_chars, Name))) # show the matching forbidden character, '.' is included
        else:
            return True
        
    # Function containg a while loop to validate a name
    def validity(self, Topic = '', Suffix_list = []):   # Suffix_list is the list of suffix which are used for testing validity of a file
        text = f'Please indicate {Topic}, without extension. Avoid special characters, only number and letters.\
            \n\n---> '
        text = cli.DisplayText(text)
        while True: # while loop to valid the project name
            name = input(text.simple())
            test = self.is_valid_filename(name) # test for valid name, return True if valid or the wrong character in the project name if not valid
            if name == '':
                invalid = 'You did no write any characters, indicate at least one letter.'
                invalid = cli.DisplayText(invalid)
                print(invalid.simple())
                input('\n\nPress ENTER to continue... ')
            else:
                if test == True:
                    self.filename = name
                    if os.path.isfile(os.path.join(self.folder, self.filename + Suffix_list[0] + self.extension)):
                        self.path = os.path.join(self.folder, self.filename + self.extension)
                        name = self.handle_existing_file()
                        break
                    else:
                        break
                else:
                    invalid = f'Invalid name, avoid using the following character(s): {test}'
                    invalid = cli.DisplayText(invalid)
                    print(invalid.simple())
                    input('\n\nPress ENTER to continue... ')
        return name

class Project():
    """
    The project object is created when it is a new project, loaded when data are processed from a pre-existing project,
    and have multiple purpose.
    First, it stores all the spectra informations created in cleaning.Spectra(). Meaning even if it is not denoised, or
    a user does not want to denoise, spectra information have to be taken. 
    It can facilitate to do homemade ANN, while it prevent opening each time mzml files when training the model or display results.
    Second, because it has all spectra information it can plot the noise traces and noise peaks.
    """
    def __init__(self, name, Folder, Newproject = True):
        if Newproject == True: # If it is a new project, we ask for saving it to a folder
            self.log = ''
            self.project_file = True # attribute to test project file
            # Project settings
            self.dir = os.path.abspath(Folder)
            self.name = name            
            self.path = os.path.join(self.dir, name + '.masslearn')             
            # Project related subfolders path
            self.featurepath = None
            self.plotpath = None            
            self.mzml_folder_path = None
            self.raw_folder_path = None # Raw files path, can be outside of the project dir
            self.raw_files_path = []
            self.mzml_files_path = [] # all mzml files path related to this project
            self.sample_names = []
            # Parameters to convert the files and denoise them
            self.rt_range = None
            self.noise_trace_threshold = None
            self.ms1_noise = None # noise count threshold
            self.ms2_noise = None
            # Parameters for MZmine and NeatMS
            self.featurelist = {} # key are sub projects names (feature list), values list of feature lists names generated via mzmine, they are not already combined
            self.files_spectra = {} # Dict of cleaning.spectra object, based on files names in self.files_path 
            self.batch = {} # this is to store the path of .xml batch files which are in /feature
            self.neatMS = None # True or false
            self.msn_df = None
            self.msn_df_path = None
            self.msn_df_deblanked = None
            self.msn_df_deblanked_path = None
            # Parameters for template and meta analysis
            self.meta = None
            self.template_path = {}
            self.template = {}
            self.template_esi_mode = {}
            self.tables_list = None
            self.label_tables_list = None
            self.treatment = None
            # Parameter for finalizing the project
            self.experiments = None
            self.experiment_FG = None
            self.complete = False
            
            
        else:
            text = 'Please indicate the relative or absolute path to your project file,\
                \nexample of absolute path: "D:\Data\myproject.masslearn" (tip: copy from adress bar in your window explorer)\
                \nIf you want to cancel this step, enter q.\
                \n\n---> '
            text = cli.DisplayText(text)            
            exist_file = False
            while exist_file == False:                                
                Projectpath = input(text.simple())
                if Projectpath.lower() == 'q':
                    self.project_file = False
                    break
                elif os.path.isfile(Projectpath + '.masslearn') or os.path.isfile(Projectpath):
                    Projectpath = os.path.splitext(Projectpath)[0]
                    with lz4.frame.open(Projectpath + '.masslearn', "rb") as file:
                        loaded_project = pickle.load(file)
                    try:
                        loaded_project.project_file     # test if attribute priject_file is present in the project file as True
                    except AttributeError:
                        error = 'This is not a valid MassLearn project file.'
                        error = cli.DisplayText(error)
                        error.warning()
                    else:
                        exist_file = True # This is a masslearn project file
            if Projectpath.lower() != 'q':
                self.name = loaded_project.name 
                self.folder = loaded_project.folder       
                self.path = loaded_project.path
                self.mzmlpath = loaded_project.mzmlpath
                self.featurepath = loaded_project.featurepath
                self.plotpath = loaded_project.plotpath
                self.files_spectra = loaded_project.files_spectra
                self.files_path = loaded_project.files_path 
                self.featurelist = loaded_project.featurelist
                self.project_file = True
                self.log = loaded_project.log
                self.noise_trace_threshold = loaded_project.noise_trace_threshold 
                self.rt_range = loaded_project.rt_range 
                self.ms1_noise = loaded_project.ms1_noise 
                self.ms2_noise = loaded_project.ms2_noise 
    
    # Function to add a featurelist to the feature list dictionnary. For each, it creates a sub dictionnary where all concerned feature lists are stored and manages with sames keys
    def add(self, Featurelistname): # Categorie is the type of list, they are kyes from the dictionnary
        pathname = os.path.join(self.featurepath, Featurelistname)
        self.featurelist[Featurelistname] = {'ms1':f'{pathname}_ms1.csv',
                                             'ms1_ann':f'{pathname}_ms1_adjusted.csv',                                             
                                             'ms2':f'{pathname}_ms2.csv',
                                             'ms2_ann':f'{pathname}_ms2_ann.csv',
                                             'msn_ann':f'{pathname}_msn.csv',
                                             'msn_table':[f'{pathname}_msn_ann_db_height.csv', f'{pathname}_msn_ann_db_area.csv']} 
    
    # Function to add a featurelist to the feature list dictionnary. For each, it creates a sub dictionnary where all concerned feature lists are stored and manages with sames keys
    def add_dash(self, Exp_title, Featurelist_path): # Categorie is the type of list, they are kyes from the dictionnary
        self.featurelist[Exp_title] = Featurelist_path
    
    # Function to update project's internal log and session log
    def update_log(self, Log):
        self.log += Log
            
    # Function to save project regurlarly on the disk at the defined path in self.path
    def save(self):
        with lz4.frame.open(self.path, "wb") as file:
            try:
                pickle.dump(self, file)
                return True
            except pickle.PicklingError as error:
                text = f'An error occurred while saving the project: {error}'
                text = cli.DisplayText(text)
                print(text.simple())
                return False
        
                
            


        
        
        
        
        
        
        
        