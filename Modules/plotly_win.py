# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:10:45 2023

@author: Ronan
"""

import os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# load iris dataset as an example
iris = load_iris()
data = iris.data
target = iris.target

# apply PCA to iris dataset
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data)

# create a dataframe from PCA result
df = pd.DataFrame(data = pca_result, columns = ['pc1', 'pc2', 'pc3'])
df['y'] = target

# create 3d plot
fig = go.Figure(data=[go.Scatter3d(
    x=df['pc1'],
    y=df['pc2'],
    z=df['pc3'],
    mode='markers',
    marker=dict(
        size=12,
        color=df['y'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# set layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# create html file
fig.write_html("temp_plot.html")

# PyQt application
app = QApplication(sys.argv)

# WebEngine View
view = QWebEngineView()
# Load the local HTML file


# Create a full path to the HTML file
full_path = os.path.abspath("temp_plot.html")
# Load the local HTML file
view.load(QUrl.fromLocalFile(full_path))


view.show()

# run application
sys.exit(app.exec_())
