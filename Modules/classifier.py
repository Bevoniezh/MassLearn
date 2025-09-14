# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:22:53 2023

@author: Ronan
"""
feat = sp.Filter('../feature/fl_ms1_h.csv')
fl = feat.featurelist
feat2 = sp.Filter('../feature/fl_ms2_h.csv')
fl2 = feat2.featurelist
# Create a new column 'ms_level' in df1 and set all values to 'ms1'
fl['ms_level'] = 'ms1'

# Create a new column 'ms_level' in df2 and set all values to 'ms2'
fl2['ms_level'] = 'ms2'

feature = pd.concat([fl, fl2], ignore_index=True)

import matplotlib.pyplot as plt
from ipywidgets import interact_manual
import pandas as pd
import numpy as np

# Define your class
class TrainingTool:
    def __init__(self, data):
        self.data = data
        self.labels = []
        self.current_index = 0

    # Function to plot the shapes of the MS1 and MS2 features
    def plot_features(self, ms1_shape, ms2_shape):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].plot(ms1_shape)
        axs[0].set_title('MS1')
        axs[0].set_xlabel('m/z')
        axs[0].set_ylabel('Intensity')

        axs[1].plot(ms2_shape)
        axs[1].set_title('MS2')
        axs[1].set_xlabel('m/z')
        axs[1].set_ylabel('Intensity')

        plt.show()

    # Function to handle the button click
    def on_button_click(self, label):
        self.labels.append(label)
        self.current_index += 1
        self.display_next_pair()

    # Function to display the next pair
    def display_next_pair(self):
        if self.current_index >= len(self.data):
            print("All pairs have been labeled.")
            return

        pair = self.data.iloc[self.current_index]

        ms1_shape = ...  # Fetch shape data for MS1 feature
        ms2_shape = ...  # Fetch shape data for MS2 feature

        print(f"MS1 Info: {pair['ms1_info']}")
        print(f"MS2 Info: {pair['ms2_info']}")
        self.plot_features(ms1_shape, ms2_shape)

        interact_manual(self.on_button_click, label=['Yes', 'No', 'Skip'])