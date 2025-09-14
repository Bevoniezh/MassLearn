# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:54:36 2023

@author: Ronan
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import Modules.file_manager as fmanager
from PyQt5.QtCore import QUrl
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from PyQt5.QtWidgets import QApplication
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWebEngineWidgets import QWebEngineView

class Filter():
    """
    This class has multiple purposes First, it is made to filter out unwanted sampels with a given pattern in their name, or keeping only them.
    Useful to filter out Pool for example.      Use self.pattern()
    Second, it remove unwanted labels from samples names.       Use self.label()
    Third, it substract the blank signal from the samples.      Uses self.blank()
    
    So:
    filtered = sp.Filter(Featurelist)
    filtered.label() # optional
    fitered.pattern() # optional    
    filtered.blank('Blank')
    
    """
    def __init__(self, Featurelistpath): 
        self.path = Featurelistpath
        self.name = os.path.splitext(os.path.basename(Featurelistpath))[0]
        self.featurelist = pd.read_csv(Featurelistpath)        
        self.height = self.featurelist.pivot_table(index=['m/z', 'rt', 'ANN'], columns='sample', values='height').fillna(0) # take height values as values for each sample column
        self.area = self.featurelist.pivot_table(index=['m/z', 'rt', 'ANN'], columns='sample', values='area').fillna(0)
        self.table = self.height.astype(str).combine(self.area.astype(str), lambda s1, s2: s1 + ',' + s2)
        
    # Function to excude from the sample name unwanted labels
    def label(self, Label): # Label is the list of labels you want to exclude from a sample name. E.g 01122020PROJECT1_sample45_rep4, ['01122020PROJECT1_', '_rep4'] will remove those two labels is they are present in a sample name   
        # Make the filtering:    
        self.height.columns = self.height.columns.str.replace(Label, '')
        self.area.columns = self.area.columns.str.replace(Label, '')
        self.table = self.height.astype(str).combine(self.area.astype(str), lambda s1, s2: s1 + ',' + s2) # this combined table is made as last step
    
    # Function to filter based on one or multiple pattern. It takes the pattern in Patternlist and filter in or out the samples based on it    
    def pattern(self, Pattern, Keep = False):    
        # Make the filtering:
        pattern = Pattern.split(',')
        cols_selected = []
        for i in pattern:
            cols = [col for col in self.height.columns if i in col and col not in cols_selected]
            cols_selected.extend(cols)
        if Keep == True:
            self.height = self.height[cols_selected]
            self.area = self.area[cols_selected]
        else:
            self.height = self.height.drop(cols_selected, axis=1)
            self.area = self.area.drop(cols_selected, axis=1)
        self.table = self.height.astype(str).combine(self.area.astype(str), lambda s1, s2: s1 + ',' + s2) # this combined table is made as last step
    
    # Function to remove a blank signal on a specific part of samples or on all samples
    def blank(self, Blankpattern = 'blank', Level = 5):        
        tables = []
        for table in [self.height, self.area]:
            # Define table_blank and table_sample which are sub-tables of samples with their linked blank column(s)
            if type(Blankpattern) == str: # if we have a list of blank sample names rather than a pattern to match
                table_b = table.filter(regex=f'{Blankpattern}.*') # get the 'blank' columns where the names contain a date 
                if (len(table_b.columns) == len(table.columns)) or table_b.empty:
                    return False # if same lenght of columns or nothing, it means there were an error in indicating blank, because it matches all columns
                colblank = table_b.columns
                table_blank = table_b.replace(0, np.nan) # replace by np.nan to not be taken in account in the calcul of the mean at next line
                table_blank['mean_blank'] =  table_blank.mean(axis=1, skipna=True) # take the mean of the blanks
                table_sample = table.drop(columns=colblank, errors='ignore') # take all columns exept them from blank table
                condition = table_sample <= Level * table_blank['mean_blank'].values[:, np.newaxis] # Define the condition for filtering, Level = 5 times higher for example
                table_sample = table_sample.mask(condition, 0) # Apply the condition to filter and update values
                tables.append(table_sample)
            else:
                df = Blankpattern
                df = df.loc[df['blank'] != '']
                sub_tables = []
                for blank_group in df['blank'].unique():
                    if blank_group != 'No_blank':
                        blank_list = blank_group.split(',')
                        sample_list = df.loc[df['blank'] == blank_group, 'sample'].to_list()
                        table_blank = table[blank_list]
                        table_blank = table_blank.replace(0, np.nan)
                        table_blank['mean_blank'] =  table_blank.mean(axis=1, skipna=True) 
                        table_sample = table[sample_list]
                        condition = table_sample <= Level * table_blank['mean_blank'].values[:, np.newaxis] 
                        table_sample = table_sample.mask(condition, 0) # Apply the condition to filter and update values
                    else:
                        sample_list = df.loc[df['blank'] == 'No_blank', 'sample'].to_list()
                        table_sample = table[sample_list] 
                    sub_tables.append(table_sample) # When No_blank there is no changes in the values because no features detected in blanks
                    
                all_sub_tables = pd.concat(sub_tables, axis=1)    
                tables.append(all_sub_tables)
        tables[0] = tables[0][~(tables[0] == 0).all(axis=1)] # remove rows with only 0s
        tables[1] = tables[1][~(tables[1] == 0).all(axis=1)] # remove rows with only 0s
        self.height = tables[0]
        self.area = tables[1]
        self.table = self.height.astype(str).combine(self.area.astype(str), lambda s1, s2: s1 + ',' + s2) # this combined table is made as last step
        
    
    # Function to write the feature list table without blanks
    def export(self):
        # TODO option to use for export self.table = self.height.astype(str).combine(self.area.astype(str), lambda s1, s2: s1 + ',' + s2)
        h_path= f'{os.path.splitext(self.path)[0]}_db_height.csv'
        self.height.to_csv(h_path) # export deblanked feature list to /feature
        a_path = f'{os.path.splitext(self.path)[0]}_db_area.csv'
        self.area.to_csv(a_path)
        
        
class Pca():
    """
    This class take as input one or multiple tables prepared with Filter() class, and combine those tables features based on rt and m/z.
    Filtering based on rt and m/z is possible.
    self.PCA('h' or 'a', 'a title') function displays a 3D PCA.
    """
    def __init__(self, Table):
        self.table = Table
        self.table.index = self.table.index.set_levels(self.table.index.levels[0].astype(float), level='m/z') # convert str() mass values to float
        self.table.index = self.table.index.set_levels(self.table.index.levels[1].astype(float), level='rt')
        self.table_f = self.table # table_f is for tabls filtered by rt and / or m/z
        self.tableT = None
              
    def tuple_to_hex(self, color_tuple):
        return "#{:02x}{:02x}{:02x}".format(int(color_tuple[0] * 255), 
                                            int(color_tuple[1] * 255), 
                                            int(color_tuple[2] * 255))
    def generate_colors(self, num_colors):
        colormap = plt.cm.tab20 # or any other colormap like "viridis", "plasma", "inferno", "magma", "cividis", etc.
        colors = [self.tuple_to_hex(colormap(i)) for i in range(colormap.N)]
        if num_colors <= len(colors):
            return colors[:num_colors]
        else:
            return colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]
        
    def find_matching_row(self, df, mz, rt, mz_tol=0.005, rt_tol=0.05): #TODO function to merge featurelists
        """Find row in df that matches the given mz and rt within certain tolerances."""
        matching_rows = df[(df.index.get_level_values('m/z').between(mz - mz_tol, mz + mz_tol)) &
                           (df.index.get_level_values('rt').between(rt - rt_tol, rt + rt_tol))]
    
        if len(matching_rows) > 0:
            return matching_rows.index[0]  # Return the first match
        else:
            return None
    """
    merged_df = pd.DataFrame()
    
    for df in df_list:
        for col in df.columns:
            if col not in merged_df.columns:
                merged_df[col] = 0.0
        
        for row in df.itertuples():
            mz, rt = row.Index
            match = find_matching_row(merged_df, mz, rt)
            if match:
                for col in df.columns:
                    if merged_df.at[match, col] == 0:
                        merged_df.at[match, col] = row._asdict()[col]
                    else:
                        merged_df.at[match, col] = str(merged_df.at[match, col]) + ',' + str(row._asdict()[col])
            else:
                new_row = pd.Series({col: row._asdict()[col] for col in df.columns}, name=(mz, rt))
                merged_df = merged_df.append(new_row)   
    """
    # Function to filter based on m/z            
    def mz_range(self, Min = '', Max = ''):
        if Min == '' and Max == '':
            pass
        elif Min == '' and Max != '':
            self.table_f = self.table_f.loc[self.table_f.index.get_level_values('m/z') <= float(Max)]
        elif Min != '' and Max == '':
            self.table_f = self.table_f.loc[self.table_f.index.get_level_values('m/z') >= float(Min)]
        else:
            self.table_f = self.table_f.loc[(self.table_f.index.get_level_values('m/z') >= float(Min)) & (self.table_f.index.get_level_values('m/z') <= float(Max))]

    
    def rt_range(self, Min = '', Max = ''):
        if Min == '' and Max == '':
            pass
        elif Min == '' and Max != '':
            self.table_f = self.table_f.loc[self.table_f.index.get_level_values('rt') <= float(Max)]
        elif Min != '' and Max == '':
            self.table_f = self.table_f.loc[self.table_f.index.get_level_values('rt') >= float(Min)]
        else:
            self.table_f = self.table_f.loc[(self.table_f.index.get_level_values('rt') >= float(Min)) & (self.table_f.index.get_level_values('rt') <= float(Max))]

    
    # Function to make a 3D PCA on an external window
    def PCA(self, Table, Title): # Valuetype is to choose feature area or intensity height
        table = Table # can be self.table or self.tableT           
        # apply PCA to iris dataset
        if 'ANN' in Table.index.names:
            labels = table.index.get_level_values('ANN')
        else:
            labels = table.index.get_level_values('label')
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(table)
        
        # create a dataframe from PCA result
        df = pd.DataFrame(data = pca_result, columns = ['pc1', 'pc2', 'pc3'])
        df['y'] = labels
        
        # add a new column with the index names from the original dataframe
        indexlist = [[str(x)for x in j]  for j in table.index.to_list() ]
        df['index_name'] = [' '.join(i[:-1]) for i in indexlist]

        
        # create color scale based on unique labels
       
        num_labels = len(np.unique(labels))
        colorscale = px.colors.qualitative.Plotly if num_labels <= 10 else self.generate_colors(num_labels)
        
        # assign color for each unique label
        colors_dict = {label:colorscale[i%len(colorscale)] for i, label in enumerate(np.unique(labels))}
        df['color'] = df['y'].map(colors_dict)
        # create 3d plot
        scatter = go.Scatter3d(
            x=df['pc1'],
            y=df['pc2'],
            z=df['pc3'],
            mode='markers',
            marker=dict(
                size=12,
                color=df['color'], # use color column
                opacity=0.8
            ),
            text=df['index_name'],  # use the index_name column for the hover text
            hoverinfo='text',
            showlegend=False
        )
        # create legend items
        legend_items = [go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=10, color=color), name=label) for label, color in colors_dict.items()]
        fig = go.Figure(data=[scatter] + legend_items)
        
        # set layout with explained variance ratio and axis labels
        fig.update_layout(
            scene = dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
            ),
            title=f'PCA Analysis - {Title}',
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        
        # WebEngine View
        view = QWebEngineView()
        # Create a full path to the HTML file        
        fig.write_html(Title + '.html')
        # Load the local HTML file
        view.load(QUrl.fromLocalFile(Title + '.html'))
        view.show()
        return view

    
    # Function to transpose data, now samples are the sites, and compounds the variables
    def transpose(self):
        self.tableT = self.table_f.copy()
        self.tableT.reset_index(inplace=True) # Reset index to get 'm/z', 'rt', 'label' back as columns
        self.tableT['compound'] = self.tableT['m/z'].astype(str) + '_' + self.tableT['rt'].astype(str) + '_' + self.tableT['ANN'].astype(str) # Convert columns to string and concatenate with '_'
        self.tableT.set_index('compound', inplace=True) # Set 'compound' as new index
        self.tableT.drop(['rt', 'm/z', 'ANN'], axis=1, inplace=True)
        self.tableT = self.tableT.transpose() # Transpose the DataFrame
        self.tableT.index.name = 'sample'
         


        
"""

############################ 2D PCA part ###################################

# 1- Simple PCA showing scatter plot
# create PCA instances for each dataframe
pca_height = PCA()
pca_area = PCA()

# fit the models to the data
pca_height.fit(df_height)
pca_area.fit(df_area)

# transform the data using the models
transformed_height = pca_height.transform(df_height)
transformed_area = pca_area.transform(df_area)

# plot PCA for df_height
plt.scatter(transformed_height[:, 0], transformed_height[:, 1])
plt.xlabel(f'PC1 ({pca_height.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_height.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA plot for heights')
plt.show()

# plot PCA for df_area
plt.scatter(transformed_area[:, 0], transformed_area[:, 1])
plt.xlabel(f'PC1 ({pca_area.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_area.explained_variance_ratio_[1]*100:.2f}%)')
plt.title('PCA plot for areas')
plt.show()


###################

# 2- PCA with vectors (samples) and labelled dots (sites, or rows). Possibility to group with colors

# Function to sort the solumns based of if there is an ID at the end
def sort_key(col_name):
    return int(col_name.split('_')[-1])
sorted_cols = sorted(df_height.columns, key=sort_key) # sort the columns based on the number at the suffix after '_'

bmi = pd.read_csv(os.path.join('..','..', '..','ABABA','BMI.csv')) # import metadata
metadata_df = pd.DataFrame({'sample': df_height.columns, 'weight_gain': bmi['Diff'].tolist()}) # create a metadata dataframe with weight gain information

# define a function to create a PCA plot
def create_pca_plot(df, metadata, filename):
    # separate the data based on weight gain
    group1 = metadata[metadata['weight_gain'] > 4.7]['sample']
    group2 = metadata[metadata['weight_gain'] <= 4.7]['sample']

    # perform PCA on the data
    pca = PCA(n_components=2)
    pca.fit(df)
    transformed_data = pca.transform(df)

    # create the plot FOR GROUPED SAMPLES BASED ON METADATA
    fig, ax = plt.subplots(figsize=(10, 10)) #, dpi=500)
    #ax.scatter(transformed_data[group1.index, 0], transformed_data[group1.index, 1], s=4, label='weight gain > 4.7')
    #ax.scatter(transformed_data[group2.index, 0], transformed_data[group2.index, 1], s=4, label='weight gain <= 4.7')
    # OR FOR A NON-GROUPED:
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], s=0.2)
    #ax.scatter(np.log10(transformed_data[:, 0]), np.log10(transformed_data[:, 1]), s=0.2)
    
    # annotate the plot with sample labels
    for i, sample in enumerate(df.index):
        
        ax.annotate(i+1, (transformed_data[i, 0], transformed_data[i, 1]), fontsize=0.2) # adjust dot label size with fontsize
       # ax.annotate(i+1, (np.log10(transformed_data[i, 0]), np.log10(transformed_data[i, 1])))
    
    # plot the PCA vectors
    feature_vectors = pca.components_.T
    arrow_size, text_pos = 20.0, 8.0 # adjust arrow length with arrow_size
    for i, v in enumerate(feature_vectors):
        
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], head_width=0.5, head_length=0.6, linewidth=0.1) # adjust vector arrow size with linewidth for ex
        ax.text(v[0]*text_pos, v[1]*text_pos, df.columns[i], color='black', ha='center', va='center', fontsize=0.1) 
        #ax.text(np.log10(v[0]*text_pos), np.log10(v[1]*text_pos), df.columns[i], color='black', ha='center', va='center', fontsize= 0.2)
        ax.annotate(df.columns[i], xy=(arrow_size*v[0], arrow_size*v[1]), xytext=(arrow_size*v[0], arrow_size*v[1]), ha='center', va='center', fontsize=0.2) # adjust vector labels size with fontsize

    # set axis labels
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    
    # IF YOU NEED TO ZOOM, here it zoom in by a factor of 5 and center at 0,0
    
    #factor = 5 # change the factor
    #ax.set_xlim(-50, max(abs(transformed_data[:,0]))/factor)
    #ax.set_ylim(-50, max(abs(transformed_data[:,1]))/factor)
    
    # set logarithmic scale for x and y axes
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    # save the plot to a file
    plt.savefig(filename)

    # export the list of dot numbers and their corresponding row index names
    dot_list = pd.DataFrame({'dot': np.arange(len(df.index))+1, 'row_index': df.index.map(str)})
    dot_list.to_csv(filename.split('.')[0] + '_dots.csv', index=False)
    
    return transformed_data

# create a PCA plot for the height dataframe
pca_height = create_pca_plot(height, metadata_df,  'height_pca_plot.png')

# create a PCA plot for the area dataframe
pca_area = create_pca_plot(area, metadata_df, 'area_pca_plot.png')



"""
