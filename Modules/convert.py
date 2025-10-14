# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:19:46 2023

@author: Ronan
"""
import os
import re
import time
import subprocess
import Modules.CLI_text as cli
from Modules.file_manager import MassLearn_directory as MLD

class RawToMZml():
    """
    Class to convert proprietary vendor files to mzML files using ProteoWizard.
    Have to be used as follow:
        raw_to_convert = convert.RawToMZml(abs/rel_path, min_elution_time, max_elution_time, file_type)
        raw_to_convert.convert_file(Log, MsconvertPath) # convert AND adjust scans of the file when required
    """
    def __init__(self, Raw_files_list, Mini = 50, Maxi = 360, File_type = 'waters'):
        self.mini = str(Mini) # minimum elution time (default 50 sec)
        self.maxi = str(Maxi) # maximum elution time (default 360 sec)
        self.raw_files = Raw_files_list # it must be absolute paths list
        self.file_type = File_type.lower() if File_type else 'waters'
        self.begin = time.time()
        
    # Function to adjust the scan number and put them in ascending order, have to be used afer convert_file()
    def adjust(self, Log, File): # do both tasks to avoid opening the File two times (it can be almost 20 s to open)
        file, _ = os.path.splitext(File)
        mzml = file + '.mzML'
        with open(mzml, 'r') as f:
            # Open a new file for writing
            with open(mzml[:-5]+'_adjusted.mzML', 'w') as output_file:                
                count_line = 0 # Iterate over each line in the file
                ref_line = -1 # This oject will help us not to change the lines with "ms level"
                for line in f:
                   # Use a regular expression to check if the line matches the pattern "spectrum index=" and contains "scan="
                    if re.search(r'spectrum index="(\d+)"', line) and "scan=" in line:
                        spectrum_index = int(re.search(r'spectrum index="(\d+)"', line).group(1)) # Extract the number following "spectrum index="
                        new_scan = spectrum_index + 1 # Calculate the new value of "scan=" as spectrum_index + 1
                        new_line = re.sub(r'scan=\d+', f'scan={new_scan}', line) # Replace the current value of "scan=" with the new value
                        # after, we adjust the ms level
                        ms_level = int(re.search(r'id="function=(\d+)', line).group(1)) # Find the true ms level based on the func data file name from raw files
                        ref_line = count_line + 2
                        output_file.write(new_line) # Write the updated line to the output file                        
                    else:
                        if count_line == ref_line:
                             new_line = re.sub(r'value="(\d+)"', f'value="{ms_level}"', line)
                             output_file.write(new_line) 
                        else:
                            output_file.write(line) # If the line doesn't match the pattern, write it to the output file unchanged                        
                    count_line += 1
        os.remove(mzml)
        os.rename(mzml[:-5]+'_adjusted.mzML', mzml) # we remove the temporary "_adjusted" and keep only .mzML


    # This funciton use msconvert to generate mzML files, but the ouput files are not correct concerning the scan numbers, you need after to use cuntion adjust()
    def convert_file(self, Log, MSconvert_path):
        # 1- Delete all func3 (meaning the lockspray signal) from raw files:        
        prefix = "_FUNC003" # Define the prefix to look for in file names
        if self.file_type == 'waters':
            for raw in self.raw_files:
                if os.path.isdir(raw):
                    for file in os.listdir(raw): # Iterate through all subfolders and remove files with the prefix
                        if file.startswith(prefix):
                            os.remove(os.path.join(raw, file)) # remove _FUNC003 definitively
                            # TODO: Indicate in MassLEarn it removes definitively func003

        # 2- Take MSconvert
        proteowizard = MSconvert_path #TODO adapt also for linux
        scantime = f"scanTime [{self.mini},{self.maxi}]"
        if self.raw_files:
            first_parent = os.path.dirname(self.raw_files[0])
            if first_parent:
                os.chdir(first_parent) # redirect the working directory to the folder where there are your raw file, for the subprocess to generate the mzML files in the right folder
        if self.raw_files != []:
            for raw_file in self.raw_files:
                cmd = [
                        proteowizard,
                        "--32",
                        "--filter",
                        "msLevel 1-2",
                        "--filter",
                        scantime,
                        "--filter",
                        "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> File:\"\"\"^<SourcePath^>\"\"\", NativeID:\"\"\"^<Id^>\"\"\"",
                        raw_file,
                        ]
                try:
                    subprocess.run(cmd, shell=False, check=True) # Run a simple command to list files in the current directory
                    os.chdir(MLD) # os path is changed for the log file
                    log = f'{os.path.basename(raw_file)} converted, time to convert: {int(time.time()-self.begin)} s'
                    Log.update(log)
                    if self.raw_files:
                        first_parent = os.path.dirname(self.raw_files[0])
                        if first_parent:
                            os.chdir(first_parent)
                    if self.file_type == 'waters':
                        self.adjust(Log, raw_file) # adjust the scan numbers
                except Exception as e:
                    f'Error to convert file: {raw_file}'
                # TODO: change the msconvert code to be compatible for Wind AND linux!

        os.chdir(MLD) # Reset the working directory

 
            


    
