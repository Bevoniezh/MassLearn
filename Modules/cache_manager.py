# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:17:03 2023

@author: Ronan
"""

import os
import fnmatch
import configparser
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath
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
    """Utility class to retrieve external software paths for the Dash app.

    Access paths with :meth:`get_path`, e.g. ``Software_DashApp().get_path('MZmine')``.
    Legacy code that indexes :attr:`path` directly continues to work.
    """

    #SOFTWARE_ENTRIES = [
    #    ("SeeMS", "seems.exe"),
    #    ("MZmine", "MZmine.exe"),
    #    ("ProteoWizard", "msconvert.exe"),
    #]

    SOFTWARE_ENTRIES = [("MZmine", "MZmine.exe")]

    def __init__(self, config_path=None):
        self.path = {}
        self._path_lookup = {}
        config_path = self._resolve_config_path(config_path)

        if not self._load_from_config(config_path):
            self._load_from_legacy_file()
        self._refresh_lookup()

    def _resolve_config_path(self, config_path):
        if config_path is not None:
            return Path(config_path)
        base_dir = Path(__file__).resolve().parent.parent
        return base_dir / "config" / "config.ini"

    def _load_from_config(self, config_path: Path) -> bool:
        parser = configparser.ConfigParser()
        read_files = parser.read(config_path, encoding="utf-8")

        if not read_files:
            return False

        section_name = next((sec for sec in parser.sections() if sec.lower() == "software"), None)
        if section_name is None:
            return False

        for software_name, executable in self.SOFTWARE_ENTRIES:
            raw_value = parser.get(section_name, software_name, fallback="").strip()
            if not raw_value:
                raw_value = parser.get(section_name, software_name.lower(), fallback="").strip()
            raw_value = raw_value.strip('"').strip("'")
            resolved = self._normalise_path(raw_value, config_path.parent, executable)
            self.path[software_name] = resolved

        return True

    def _load_from_legacy_file(self):
        #msconvert = ['ProteoWizard', 'msconvert.exe']
        mzmine = ['MZmine', 'MZmine.exe']
        #seems = ['SeeMS', 'seems.exe']
        legacy_path = Path('./data/software_path_dash.dat')
        soft_list = [mzmine]
        for software, _ in self.SOFTWARE_ENTRIES:
            self.path.setdefault(software, '')

        if not legacy_path.exists():
            return

        with legacy_path.open('r+') as f:
            file = f.readlines()
            for software, line in zip(soft_list, file):
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
                self.path[software[0]] = line.split(' # ')[1].strip()
        self._refresh_lookup()

    def _normalise_path(self, raw_value: str, base_dir: Path, executable: str) -> str:
        if not raw_value:
            return ''

        expanded = os.path.expanduser(raw_value)
        if self._is_windows_absolute(expanded):
            return self._normalise_windows_path(expanded, executable)

        path_obj = Path(expanded).expanduser()
        if not path_obj.is_absolute():
            path_obj = (base_dir / path_obj).resolve(strict=False)
        else:
            path_obj = path_obj.resolve(strict=False)

        if path_obj.is_dir():
            final_path = path_obj / executable
        elif path_obj.suffix or path_obj.name.lower().endswith(executable.lower()):
            final_path = path_obj
        else:
            final_path = path_obj / executable

        return os.path.normpath(str(final_path))

    @staticmethod
    def _is_windows_absolute(path_str: str) -> bool:
        return bool(re.match(r'^[A-Za-z]:[\\/]', path_str)) or path_str.startswith('\\\\')

    @staticmethod
    def _normalise_windows_path(path_str: str, executable: str) -> str:
        windows_path = PureWindowsPath(path_str)
        lower_path = str(windows_path).lower()

        if lower_path.endswith(executable.lower()) or windows_path.suffix:
            return str(windows_path)

        return str(windows_path / executable)

    def _refresh_lookup(self):
        self._path_lookup = {name.lower(): value for name, value in self.path.items()}

    def get_path(self, software_name: str, default: str = '') -> str:
        """Return the configured path for a software entry.

        Parameters
        ----------
        software_name:
            Name of the software as declared in :attr:`SOFTWARE_ENTRIES`. The lookup is
            case-insensitive and accepts either ``"MZmine"`` or ``"mzmine"``.
        default:
            Value returned when the requested software has no configured path.
        """

        if not software_name:
            return default

        return self._path_lookup.get(software_name.lower(), default)

    def __getitem__(self, software_name: str) -> str:
        return self.get_path(software_name)

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
        