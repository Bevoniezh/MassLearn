# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:57:08 2023

@author: Ronan
"""

import os
import ezodf
import psutil
import shutil
import threading
import subprocess
import logging
import pandas as pd
from lxml import etree
from xml.etree.ElementTree import tostring
import Modules.file_manager as fmanager
import Modules.CLI_text as cli
import Modules.cache_manager as cache_manager
from Modules import logging_config

logger = logging.getLogger(__name__)


class Batch():
    """
    This class manage everything related to mzmine XML batch files.
    It takes batch_default.xml from Cache folder as default file, and modify with the values from a parameters.ods file.
    A user can create a personnalized batch using MZmine, batch class will integrate it for\
    the research mode.
    """
    
    def __init__(self, Batchdefault = './Cache/batch_default_ms1.xml', Parameters = './Batch_files/parameters.ods'):
        
        self.batchdefault = Batchdefault    
        
        # Import ods file and convert in a suitable df
        print(Parameters)
        ods_file = ezodf.opendoc(Parameters) # open ods document
        sheet = ods_file.sheets[0]
        data = []
        for row in sheet.rows():
            data.append([cell.value for cell in row])
        df = pd.DataFrame(data)
        df = df.drop(columns=df.columns[-1]) # remove last column which is an empty one
        df.columns = df.iloc[0,:] # change col names takin the first row as header
        df = df.drop(0) #remove the first row because now it is the header       
        df = df.fillna('nan')
        df.index = range(len(df))
        
        self.parameters = df 
        self.parser = etree.XMLParser(remove_blank_text=True)
        self.tree = None
        self.root = None
        self.MassLearn_directory = os.getcwd()
        
        
    # Function to create a batch from the parameters defined in parameters.ods and taking as reference batch batch_default.xml in the cache folder
    def create_batch(self, Newfilename = '', Folder = '', Dash = False): #TODO all created batch in a project can be saved in the project file
        self.tree = etree.parse(self.batchdefault, self.parser)
        self.root = self.tree.getroot()    
        ods = self.parameters
        batchsteps = {}
        for row in range(len(ods)):
            if 'batchstep' == ods.loc[row, 'TAG']: # Count all the modules and there line number from parameters.ods
                module = ods.loc[row, 'NAME']     
                if module not in batchsteps.keys():
                    batchsteps[module] = [row] # A dictionnary is used to store the module line index (meaning dataframe row) of every unique module occurence. Typically, it can have two mass detection module called
                else:
                    batchsteps[module].append(row)
        batchstep_rows = [item for sublist in batchsteps.values() for item in sublist]
        for batchstep, row_list in batchsteps.items(): # iterate throught the dictionnary keys, meaning from module type to other module type
            self.proceed_module(batchstep, row_list, batchstep_rows)
        obj = fmanager.FileManager(Newfilename, Folder, self.tree, 'xml')
        if Dash == False:
            path = obj.saving_file(handle=False)
        else:
            path = obj.saving_file_dash()
        with open(path, encoding='UTF-8') as f: # Open the file in read mode
            lines = f.readlines()
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?><batch mzmine_version="3.4.16">\n' # Replace the first line with the new line
        with open(path, "w", encoding="utf-8") as f: # Open the file in write mode and write the modified lines
            f.writelines(lines)


    # Function to modify the xml parsed object called self.root with the corresponding values from parameters.ods
    def proceed_module(self, Batchstep, Row_list, Batchstep_rows):
        ods = self.parameters  # ods is taken as variable name for better concistency       
        for count, batchstep_idx in enumerate(Row_list): # iterate trhough the row list in case there are two modules  with the same name, typically the mass detection module
            if batchstep_idx != max(Batchstep_rows): # verify if the batchstep is not the last one
                next_batchstep = next(x for x in Batchstep_rows if int(x) > batchstep_idx) # next batschstep index is taken to iterate all the parameters of the current batchstep proceeded without continuing after to the other batchstep
            else:
                next_batchstep = len(ods) # if it is the last batchstep, the iteration will finish at the last line of the parameters table
            batchstep_elements = self.root.xpath(f"//batchstep[@method='{Batchstep}']")[count]
            for param_row in range(batchstep_idx, next_batchstep): # iteration through the parameters of a batchstep
                tag = ods.loc[param_row, 'TAG'] # xml tag of the parameter
                element_name = ods.loc[param_row, 'NAME'] # name of the tag after name=
                element_tochange = ods.loc[param_row, 'ELEMENT'] # the lxml element to be changed, can be text between tags, or selected item in a tag               
                value = str(ods.loc[param_row, 'VALUE_NORMAL_MODE']) # The new value
                if element_tochange == 'nan': # For all batchstep and parameter that need no change but are here for the program running
                    pass
                elif tag == 'file': # special tags file are proceed elsewhere after this condition 
                    pass
                elif ods.loc[param_row, 'PARENT'] != 'nan' and element_name != 'nan':
                    parent_tag, parent_name = ods.loc[param_row, 'PARENT'].split(',') # to identify the parent tag for more precision, it can happen a specific tag to appear multiple times in a same batchstep, this prevent unwanted parameter to be modified
                    element = batchstep_elements.xpath(f'//{tag}[@name="{element_name}"][ancestor::{parent_tag}[@name="{parent_name}"]]')[0]
                elif ods.loc[param_row, 'PARENT'] != 'nan' and element_name == 'nan':
                    parent_tag, parent_name = ods.loc[param_row, 'PARENT'].split(',')
                    element = batchstep_elements.xpath(f'//{tag}[ancestor::{parent_tag}[@name="{parent_name}"]]')[0]
                elif ods.loc[param_row, 'PARENT'] == 'nan' and element_name == 'nan':
                    element = batchstep_elements.xpath(f'//{tag}')[0]
                else:
                    for el_name in element_name.split(','): # it can happen to modify two different lines in the batch xml file for the same kind of parameter, typically for Retention time smoothing and width
                        element = batchstep_elements.xpath(f'//{tag}[@name="{el_name}"]')[0]
                                
                if tag == 'file': # in tag file, it can have multiple file imported
                    parent_tag, parent_name = ods.loc[param_row, 'PARENT'].split(',') #all corrresponding name or pathname of samples files are separated and stored in a list
                    batchstep_subelements = batchstep_elements.xpath(f"//{parent_tag}[@name='{parent_name}']")[0]
                    all_path = value.split(',')
                    for count, path in enumerate(all_path):
                        if count == 0: # for the first file, we simply update the already existing tag
                            element = batchstep_subelements.xpath(f'//{tag}')[0]
                            element.text = os.path.abspath(path)
                        else: # for the other files, we add new tags
                            new_file_elem = etree.Element('file')
                            new_file_elem.text = os.path.abspath(path)
                            batchstep_subelements.append(new_file_elem)
                elif element_tochange == 'element.text': # modify text in element  
                    if tag == 'current_file':
                        element.text = os.path.abspath(value)
                    else:    
                        element.text = value
                elif element_tochange == "element.attrib['selected']": # modify the content of "selected" atttribute in a tag 
                    element.attrib['selected'] = value
                elif element_tochange == "element.attrib['selected_item']":
                    element.attrib['selected_item'] = value
                else: 
                    pass

            


    #Function to get back in the Batch_file folder a parameters.ods file with default parameter, taken from the cache. It will overwrite if there is a file already called parameters.ods        
    def reset_default(self, Parameters = './Batch_files/parameters.ods', Default = './Cache/parameters_default.ods'):
        shutil.copy2(Default, Parameters)
        
        


class MZmine():
    """
    This class of object manage MZmine runs
    """    
    def __init__(self, Batch, Featurelistname, Temp = '..\Temp' ):
        self.fln = Featurelistname
        self.batchpath = os.path.abspath(Batch)
        self.temp = os.path.abspath(Temp)
        self.MassLearn_directory = os.getcwd()
        self.message = ''
    
    def start_run(self):
        # This method starts the run method in a new thread
        thread = threading.Thread(target=self.run_dashmode)
        thread.start()
        return thread # Return the thread object so you can check if it's still running

    def run(self): #TODO  - linux adaptation
        self.delete_files_in_folder(self.temp)
        with open('./Cache/software_path.dat', 'r+') as fic:
            file = fic.readlines()
        for line in file:
            if 'MZmine' in line:
                mzmine_exe = line.split(' # ')[1][:-1]
        cmd = [
            f'{mzmine_exe}\MZmine.exe',
            "-batch",
            f'{self.batchpath}',
            "-memory", "none",
            "-temp",
            f'{self.temp}',
            ]
        log = f'Begin of MZmine run for {os.path.basename(self.batchpath)}. No estimation of time to the end, be patient it will not take too long, just some few minutes.'
        logging_config.log_info(logger, log)
        text = cli.DisplayText(log)
        print(text.simple())
        subprocess.run(cmd, shell=True, capture_output=True, text=True) # Run a simple command to list files in the current directory
        log = f'Feature list {os.path.basename(self.fln)} created.'
        logging_config.log_info(logger, log)
        text = cli.DisplayText(log)
        print(text.simple())

    def run_dashmode(self): #TODO  - linux adaptation
        self.delete_files_in_folder(self.temp)
        software_manager = cache_manager.Software_DashApp()
        mzmine_exe = software_manager.get_path('MZmine', '').strip()

        if mzmine_exe:
            cmd = [
                f'{mzmine_exe}',  # Make sure to adjust for Linux if needed
                "-batch",
                f'{self.batchpath}',
                "-memory", "none",
                "-temp",
                f'{self.temp}',
            ]
            expected_output = os.path.abspath(self.fln)
            log = f'Begin of MZmine run for {os.path.basename(self.batchpath)}. '
            logging_config.log_info(logger, log)

            # Run the subprocess and capture its output
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            stderr = (result.stderr or '').strip()
            if result.returncode != 0:
                log = f'Done - Error running MZmine: {stderr if stderr else "Unknown error."}'
                self.message = log
                logging_config.log_error(logger, log)
            elif not os.path.exists(expected_output):
                log = (
                    "Done - MZmine finished but the expected feature list was not created. "
                    f"Expected file: {expected_output}. Make sure MZmine is running and that you are signed in with a valid account before launching the process."
                )
                self.message = log
                logging_config.log_error(logger, log)
            else:
                log = f'Feature list {os.path.basename(self.fln)} created.'
                self.message = log
                logging_config.log_info(logger, log)

            print(log)  # Adjust according to how you want to handle logging
        else:
            log = "Done - MZmine path not found. Configure it from the software manager."
            logging_config.log_warning(logger, log)
            print(log)  # Adjust according to how you want to handle logging
            self.message = log
        return self.message
    
    # Function used to delete temp files
    def delete_files_in_folder(self, folder_path):
        # Remove the directory and all its contents
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        
        # Recreate the directory
        os.makedirs(folder_path)


#Function to get the size of the free RAM in GBytes
def RAM():
    free_memory = psutil.virtual_memory().free
    used_memory = psutil.virtual_memory().used
    return round(free_memory/(1024**3),2), round(used_memory/(1024**3),2)


