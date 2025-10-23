# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:19:46 2023

@author: Ronan
"""
import os
import re
import time
import logging
import subprocess
from typing import Dict, List, Optional
from Modules import logging_config

logger = logging.getLogger(__name__)

TITLE_TEMPLATE = (
    'titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> '
    'File:"<SourcePath>", NativeID:"<Id>"'
)

VENDOR_SETTINGS: Dict[str, Dict[str, Optional[List[str]]]] = {
    'waters': {
        'additional_filters': [],
        'extra_args': [],
        'pre_conversion': ['_remove_waters_lockspray'],
        'post_conversion': ['adjust'],
    },
    'thermo': {
        'additional_filters': [
            'peakPicking vendor 1-',
            'zeroSamples removeExtra',
        ],
        'extra_args': [],
        'pre_conversion': None,
        'post_conversion': None,
    },
    'bruker': {
        'additional_filters': [
            'combineIonMobilitySpectra',
            'peakPicking true 1-',
            'zeroSamples removeExtra',
        ],
        'extra_args': [],
        'pre_conversion': None,
        'post_conversion': None,
    },
    'sciex': {
        'additional_filters': [
            'peakPicking true 1-',
            'zeroSamples removeExtra',
        ],
        'extra_args': [],
        'pre_conversion': None,
        'post_conversion': None,
    },
}

class RawToMZml():
    """
    Class to convert proprietary vendor files to mzML files using ProteoWizard.
    Have to be used as follow:
        raw_to_convert = convert.RawToMZml(abs/rel_path, min_elution_time, max_elution_time, file_type)
        raw_to_convert.convert_file(MsconvertPath) # convert AND adjust scans of the file when required
    """
    def __init__(self, Raw_files_list, Mini = 50, Maxi = 360, File_type = 'waters'):
        self.mini = str(Mini) # minimum elution time (default 50 sec)
        self.maxi = str(Maxi) # maximum elution time (default 360 sec)
        self.raw_files = Raw_files_list # it must be absolute paths list
        self.file_type = File_type.lower() if File_type else 'waters'
        self.begin = time.time()
        
    # Function to adjust the scan number and put them in ascending order, have to be used afer convert_file()
    def adjust(self, File): # do both tasks to avoid opening the File two times (it can be almost 20 s to open)
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


    def _remove_waters_lockspray(self):
        prefix = "_FUNC003" # Define the prefix to look for in file names
        removed_files = []
        for raw in self.raw_files:
            if os.path.isdir(raw):
                for file in os.listdir(raw): # Iterate through all subfolders and remove files with the prefix
                    if file.startswith(prefix):
                        target = os.path.join(raw, file)
                        try:
                            os.remove(target) # remove _FUNC003 definitively
                            removed_files.append(target)
                        except OSError as error:
                            logging_config.log_warning(
                                logger,
                                'Unable to delete Waters lockmass file %s: %s',
                                target,
                                error,
                            )
        if removed_files:
            logging_config.log_info(
                logger,
                'Removed %d Waters lockmass files prior to conversion.',
                len(removed_files),
            )

    def _get_vendor_settings(self):
        settings = VENDOR_SETTINGS.get(self.file_type)
        if settings is None:
            logging_config.log_warning(
                logger,
                'Unknown raw file type "%s". Falling back to Waters defaults.',
                self.file_type,
            )
            settings = VENDOR_SETTINGS['waters']
        return settings

    def _build_filters(self, scantime: str, vendor_filters: List[str]) -> List[str]:
        filters = [
            'msLevel 1-2',
            scantime,
        ]
        filters.extend(vendor_filters)
        filters.append(TITLE_TEMPLATE)
        args: List[str] = []
        for filter_arg in filters:
            if filter_arg:
                args.extend(["--filter", filter_arg])
        return args

    # This funciton use msconvert to generate mzML files, but the ouput files are not correct concerning the scan numbers, you need after to use cuntion adjust()
    def convert_file(self, MSconvert_path):
        settings = self._get_vendor_settings()
        pre_conversion_steps = settings.get('pre_conversion') or []
        for step in pre_conversion_steps:
            getattr(self, step)()

        proteowizard = MSconvert_path #TODO adapt also for linux
        scantime = f"scanTime [{self.mini},{self.maxi}]"

        vendor_filters = settings.get('additional_filters') or []
        filter_args = self._build_filters(scantime, vendor_filters)
        extra_args = settings.get('extra_args') or []

        if not self.raw_files:
            return

        first_parent = os.path.dirname(self.raw_files[0])
        original_cwd = os.getcwd()
        try:
            if first_parent:
                os.chdir(first_parent) # redirect the working directory to the folder where there are your raw file, for the subprocess to generate the mzML files in the right folder

            for raw_file in self.raw_files:
                cmd = [
                        proteowizard,
                        "--mzML",
                        "--32",
                        *extra_args,
                        *filter_args,
                        raw_file,
                        ]
                try:
                    result = subprocess.run(
                        cmd,
                        shell=False,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    if result.stderr:
                        logging_config.log_warning(
                            logger,
                            'msconvert reported warnings for %s: %s',
                            os.path.basename(raw_file),
                            result.stderr.strip(),
                        )
                    logging_config.log_info(
                        logger,
                        '%s converted, time to convert: %s s',
                        os.path.basename(raw_file),
                        int(time.time()-self.begin),
                    )
                    post_steps = settings.get('post_conversion') or []
                    for step in post_steps:
                        getattr(self, step)(raw_file)
                except subprocess.CalledProcessError as error:
                    message = error.stderr or error.stdout or str(error)
                    logging_config.log_error(
                        logger,
                        'Error converting %s: %s',
                        os.path.basename(raw_file),
                        message.strip(),
                    )
                    raise
        finally:
            os.chdir(original_cwd)

 
            


    
