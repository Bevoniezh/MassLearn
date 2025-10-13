# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:17:03 2023

@author: Ronan
"""

import os
import fnmatch
from datetime import datetime
import Modules.CLI_text as cli

class Software():
    """
    Class to get the path of a software MZmine and MSconvert    
    """
    
    def __init__(self):
        self.path = {}
        msconvert = ['ProteoWizard', 'msconvert.exe']
        mzmine = ['MZmine', 'MZmine.exe'] 
        with open('./data/software_path.dat', 'r+') as f:
            file = f.readlines()
            for software, line in zip([msconvert, mzmine], file):
                if software[1] not in line:
                    text = f'Need to search for path of {software[1]}, please wait.'
                    text = cli.DisplayText(text)
                    print(text.simple())  
                    soft = FindSoftware(software)
                    path, folder_found = soft.search_software()
                    if folder_found == False:
                        text = f'{software[1]} not found. Please make sure the software is present/installed. The folder must have {software[0]} in its name and the {software[1]} file have to have to be written like this.'
                        text = cli.DisplayText(text)
                        print(text.simple())
                    else:
                        f.write(f'{software} # {path}\n')
                self.path[software[0]] = line.split(' # ')[1][:-1]
                
class Software_DashApp():
    """
    Class to get the path of a software MZmine and MSconvert    
    """
    
    def __init__(self):
        self.path = {}
        msconvert = ['ProteoWizard', 'msconvert.exe']
        mzmine = ['MZmine', 'MZmine.exe'] 
        seems = ['SeeMS', 'seems.exe']
        with open('./data/software_path_dash.dat', 'r+') as f:
            file = f.readlines()
            for software, line in zip([seems, mzmine, msconvert], file):
                if software[1] not in line:
                    text = f'Need to search for path of {software[1]}, please wait.'
                    text = cli.DisplayText(text)
                    print(text.simple())  
                    soft = FindSoftware(software)
                    path, folder_found = soft.search_software()
                    if folder_found == False:
                        text = f'{software[1]} not found. Please make sure the software is present/installed. The folder must have {software[0]} in its name and the {software[1]} file have to have to be written like this.'
                        text = cli.DisplayText(text)
                        print(text.simple())
                    else:
                        f.write(f'{software} # {path}\n')
                self.path[software[0]] = line.split(' # ')[1]

class FindSoftware():
    """
    Class of object to find the path of a executable software like MZmine.exe
    """
    def __init__(self, Software):
        self.folder = Software[0]
        self.software = Software[1]
    
    # Function to get the folder path of a software with an automatic research
    def search_software(self):
        folder_found = False
        root_path = ''  # Define the root path to start the search
        filename_pattern = self.software  # Define the filename pattern to match
        folder_pattern = self.folder.lower()  # Define the folder pattern to match (converted to lowercase)
        uniform_path = ''
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in fnmatch.filter(filenames, filename_pattern):
                # If a matching file is found, get the parent directory of the directory containing the file
                parent_dir = os.path.dirname(os.path.join(dirpath, filename))
                uniform_path = os.path.normpath(parent_dir).replace('\\', '/')
                if folder_pattern not in os.path.basename(parent_dir).lower():
                    continue
                else:
                    folder_found = True
                    break  # Stop the search
            else:
                continue  # If the inner loop doesn't break, continue to the next directory
            break  # If the inner loop breaks, break out of the outer loop as well
        return uniform_path, folder_found
    

class LogManager():
    """
    Class to manage the session log file
    """
    def __init__(self): 
        self.session = ''
        self.kill_log() # when new session of MassLearn is run, the previous log is reset
        self.user = None # is the user of the current session
        self.general = './data/log-backup' # this is a log text wich will be never deleted, until it is deleted it will store every information updated on the log of all user for all sessions
        
    def update(self, Text_input, Project = None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        with open('log.log', 'a') as file:
            file.write(f'{now} -- {Text_input}\n')     
        with open(self.general, 'a') as file:
            file.write(f'{now} -- {Text_input}\n') 
        with open(f'data/{self.user}.log', 'a') as file: # append user's log
            file.write(f'{now} -- {Text_input}\n') 
        if Project != None:
            Project.update_log(f'{now} -- {Text_input}\n')
        self.session = f'{now} -- {Text_input}\n'
        return self.session # this is for the Project log
    
    
    # To empty the log file
    def kill_log(self):
        self.session = ''
        with open('log.log', 'w') as file:
            file.write('')
        