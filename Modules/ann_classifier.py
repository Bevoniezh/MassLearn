# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:08:00 2023

@author: Ronan
"""

import pymzml
import pandas as pd

# Load the feature list from a csv file
def load_feature_list(csv_file_path):
    feature_list = pd.read_csv(csv_file_path)
    return feature_list

# Load the mzML file for a specific sample
def load_mzml_file(mzml_file_path):
    run = pymzml.run.Reader(mzml_file_path)
    return run

# Extract the relevant data for a specific feature from the mzML file
def extract_feature_data(run, feature):
    min_rt = feature['min_rt']
    max_rt = feature['max_rt']
    min_mz = feature['min_mz']
    max_mz = feature['max_mz']

    feature_data = []
    for spectrum in run:
        if spectrum['MS level'] == 1 and min_rt <= spectrum['scan time'] <= max_rt:
            for mz, intensity in spectrum.centroidedPeaks:
                if min_mz <= mz <= max_mz:
                    feature_data.append((mz, intensity))
    
    return feature_data

# Prepare the data for input to the ANN
def preprocess_data(feature_list_csv, mzml_files):
    feature_list = load_feature_list(feature_list_csv)
    data = []

    for mzml_file in mzml_files:
        run = load_mzml_file(mzml_file)
        for index, feature in feature_list.iterrows():
            feature_data = extract_feature_data(run, feature)
            # Further preprocessing might be required here to prepare the data for the ANN
            data.append(feature_data)
    
    return data
