# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:21:26 2024

@author: Ronan
"""
import os
import numpy as np
import pandas as pd


class Questions():
    def __init__(self):
        self.table = pd.read_csv('data/science_qa.csv')

        # Assuming df is your DataFrame
        self.random_row = self.table.sample(n=1)