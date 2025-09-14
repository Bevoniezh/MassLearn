# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:03:36 2023

@author: Ronan
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

libraries = [
    "re",
    "os",
    "bs4",
    "math",
    "scipy",
    "base64",
    "pymzml",
    "peakutils",
    "numpy",
    "subprocess",
    "fnmatch",
    "pandas",
    "time",
    "shutil",
    "matplotlib",
    "lxml"
]

def check_libraries():
    for module in libraries:
        try:
            __import__(module)
        except ImportError:
            print(f"{module} not found. Installing...")
            install(module)
            