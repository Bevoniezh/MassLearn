
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:16:57 2023

@author: Ronan

This file creates a Dash app. It displays multiple tools for statistical anayzes of the csv files generated via MassLearn.

"""

###############################################################################
###############################################################################
import os
import re
import json
import dash
import scipy
import logging
import warnings
import itertools
import threading
import scipy.stats
from typing import Dict, List, Optional
from contextlib import contextmanager
import numpy as np
import pandas as pd
import networkx as nx
from diskcache import Cache
import plotly.express as px
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import networkx.algorithms.community as nx_comm
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from dash.dependencies import Input, Output, State
from sklearn.cross_decomposition import PLSRegression
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from dash import html, dcc, dash_table, callback_context, callback
import lz4.frame

from Modules import logging_config

logger = logging.getLogger(__name__)

cache = Cache('./disk_cache')

# Code to suppress specific pandas warning
warnings.filterwarnings('ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*')

dash.register_page(__name__)

# Navbar definition
###############################################################################

project_loaded = cache.get('project_loaded')

if project_loaded != None:
    project_name = project_loaded.name
    Project_menu = dbc.InputGroupText(
                children=project_name,
                id="analytics-project-menu",
            ),
else:
    Project_menu = dbc.InputGroupText("Create or load a project", id="analytics-project-menu")

navbar = dbc.Navbar(
    dbc.Container([
            html.A(
            dbc.Row([
                dbc.Col(html.Img(src='/assets/logo.png', height="40px")),
                dbc.Col(
                    dbc.NavbarBrand("<- -        Main Menu", className="ms-2",
                                    style={"fontSize": "16px"})  # smaller font
                ),
            ],
            align="center",
            className="g-0",
            ),
            href="/home",
            style={"textDecoration": "none"},
        ),
             dbc.Col(
            "Analytical Dashboard",
            width="auto",
            className="d-flex justify-content-center",
            style={"fontSize": "20px", "fontWeight": "bold"}  # bigger font
             ),
            dbc.Col(
                Project_menu,
                width="auto",
                className="d-flex justify-content-right", 
            ),
    ]),
    color="dark",
    dark=True,
    style={'height': '50px'},
)


# Define functions
###############################################################################
# Applying the function to each group and creating the preprocessed_df DataFrame

# Featurelist completion
######################################
# Function to add for each feature the samples where the values are 0 (it have been renoved in msn_df)
def add_zeros(Df, Samples): # Df is featurelist, meaning msn_df_deblanked, Samples is the list of samples
    # Extract the unique features
    features = Df['feature'].unique()
    
    # Columns to copy from existing rows
    columns_to_copy = ['feature', 'rt', 'm/z', 'MS_level']
    default_values = {}
    for col in Df.columns:
        if col in columns_to_copy + ['sample', 'mzml_name']:
            continue
        if col == 'shape_id':
            default_values[col] = None
        else:
            default_values[col] = 0
    
    # Create a new dataframe to hold the complete data
    new_rows = []
    
    for feature in features:
        feature_rows = Df[Df['feature'] == feature]
        for sample in Samples:
            if sample not in feature_rows['sample'].values:
                # Find a template row to copy constant values
                template_row = feature_rows.iloc[0]
                new_row = template_row.copy(deep=True)
                new_row['sample'] = sample
                new_row['mzml_name'] = sample + ".mzML"
                for col, default in default_values.items():
                    new_row[col] = default
                new_rows.append(new_row)
    
    # Create a dataframe with the new rows and append to the original dataframe
    new_rows_df = pd.DataFrame(new_rows)
    complete_df = pd.concat([Df, new_rows_df], ignore_index=True)
    return complete_df

# Redundancy analysis
######################################
# Function to perform RDA and return R^2 values for a given predictor set
def perform_rda(predictors, features):
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(predictors, features)
    features_predicted_by_predictor = linear_regression_model.predict(predictors)
    r_squared_values_predictor = r2_score(features, features_predicted_by_predictor, multioutput='raw_values')
    return r_squared_values_predictor

# Function to perform permutation test and return p-value
def permutation_test(predictors, features, original_r_squared, n_permutations=10):
    count = 0
    for _ in range(n_permutations):
        # Permute the predictors
        shuffled_predictors = shuffle(predictors)
        # Recalculate R^2 with permuted data
        shuffled_r_squared = perform_rda(shuffled_predictors, features).mean()
        # Count how many times the permuted R^2 is greater than or equal to the original R^2
        if shuffled_r_squared >= original_r_squared:
            count += 1
    # Calculate p-value: proportion of permuted R^2 >= original R^2
    p_value = count / n_permutations
    if p_value == 0:
        p_value = '0.000'
    return p_value
    

# Function to export for RDA analysis using the provided Rmarckdown
def export_rda(Stand_ion_df):
    global levels, treatment, featurepath, featurelistname
    response = Stand_ion_df.copy(deep=True) # to defragment data. Response is the response variable df, it will be with samples as index, and feature groups as columns
    response['feature_group'] = response['FG'].astype(int).astype(str) + ' - ' + response['rt'].round(2).astype(str) + ' ' + response['m/z'].astype(str) # Combine two columns with an underscore as a delimiter to form a new index
    response.drop(['rt', 'm/z', 'Size', 'FG'], axis=1, inplace=True) # Remove the original columns used to form the index
    response = response.set_index('feature_group').T.reset_index().rename(columns={'index': 'sample'}) # as index rows 'sample', meaning all samples, as column all feature_groups or feature groups
    
    explanatory = {'sample':[]} # explanatory variable are the metadata associated (treatments) to the response for the multivariate analysis
    for s in response['sample']:
        explanatory['sample'].append(s)
        for l in levels:
            if l not in explanatory.keys():
                explanatory[l] = []
            for k, v in treatment.items():
                if k[1] == l and s in v:
                    explanatory[l].append(k[0])
    explanatory = pd.DataFrame(explanatory) # as rows the samples, as columns the associated metadata class for each samples as values the class values corresponding to the samples              
    
    response.to_csv(os.path.join(featurepath, featurelistname + '_response.csv'), index=False)
    explanatory.to_csv(os.path.join(featurepath, featurelistname + '_explanatory.csv'), index=False)

# Function to create the variation table. 
def make_variance_df(Stand_ion_df):
    global levels, treatment, featurepath, featurelistname
    response = Stand_ion_df.copy(deep=True) # to defragment data. Response is the response variable df, it will be with samples as index, and feature groups as columns
    response['feature_group'] = response['FG'].astype(int).astype(str) + ' - ' + response['rt'].round(2).astype(str) + '_' + response['m/z'].astype(str) # Combine two columns with an underscore as a delimiter to form a new index
    response.drop(['rt', 'm/z', 'Size', 'FG'], axis=1, inplace=True) # Remove the original columns used to form the index
    response = response.set_index('feature_group').T.reset_index().rename(columns={'index': 'sample'}) # as index rows 'sample', meaning all samples, as column all feature_groups or feature groups
    
    explanatory = {'sample':[]} # explanatory variable are the metadata associated (treatments) to the response for the multivariate analysis
    for s in response['sample']:
        explanatory['sample'].append(s)
        for l in levels:
            if l not in explanatory.keys():
                explanatory[l] = []
            for k, v in treatment.items():
                if k[1] == l and s in v:
                    explanatory[l].append(k[0])
    explanatory = pd.DataFrame(explanatory) # as rows the samples, as columns the associated metadata class for each samples as values the class values corresponding to the samples              

    merged_data = pd.merge(response, explanatory, on='sample') # combination of the two df, with sample as ref, so there are as many rows as samples, all feature groups as columns + all metadata class as columns
    metadata = merged_data.drop(['sample'], axis = 1) # remove sample col

    # One-hot encoding the metadata
    encoder = OneHotEncoder()
    metadata_encoded = encoder.fit_transform(metadata).toarray() # the metadata df as the same structure but now the values are 0 and 1
    
    # Creating a DataFrame for the encoded metadata with proper column names
    metadata_encoded_df = pd.DataFrame(metadata_encoded, columns=encoder.get_feature_names_out()) # col names are fature groups identified with rt_mz and also metadata classes
    
    # Isolating the LCMS features data
    col_to_del = ['sample'] + levels
    features = merged_data.drop(columns=col_to_del) # keep only features, remove metadata columns
    levels_coef_determination_dict = {}
    levels_p_value_dict = {}
    
    total_explained_variance_individual  = 0
    overall_explained_variance = 0
    for l in levels: # iterate through the metadata classes (Technician, Line, Treatment_classes, ...) and make RDA using all feature groups as varaibles 
        level_encoded = metadata_encoded_df[[col for col in metadata_encoded_df.columns if l in col]]
        
        # Perform RDA for operator and nematode separately
        r_squared_values_level = perform_rda(level_encoded, features) # R2 of the level
        
        levels_coef_determination_dict[l] = r_squared_values_level
        total_explained_variance_individual += r_squared_values_level
        overall_explained_variance += r_squared_values_level.mean()
        
        # Make permutation tests
        p_value_level = permutation_test(level_encoded, features, r_squared_values_level.mean())
        levels_p_value_dict[l] = p_value_level
        
    # Calculate the unexplained variance for each feature by subtracting the explained variance from 1
    unexplained_variance_individual = 1 - total_explained_variance_individual

    # Overall unexplained variance
    overall_unexplained_variance = 1 - overall_explained_variance
    
    variance_dict = {'FG/MFG (index min_Da)' : features.columns}
    overall_variances_dict = {"FG/MFG (index min_Da)": "Overall"}
    overall_p_value_dict = {"FG/MFG (index min_Da)": "p value"}
    for l in levels:
        variance_dict[l] = levels_coef_determination_dict[l]
        overall_variances_dict[l] = levels_coef_determination_dict[l].mean()
        overall_p_value_dict[l]  = str(levels_p_value_dict[l])
    variance_dict["Unexplained"] = unexplained_variance_individual
    overall_variances_dict["Unexplained"] = overall_unexplained_variance
    overall_p_value_dict["Unexplained"] = '  -  '
    
    # Compiling the results into a DataFrame
    variance_df_individual = pd.DataFrame(variance_dict)
    
    overall_variances_df = pd.DataFrame([overall_variances_dict])
    overall_p_value_df = pd.DataFrame([overall_p_value_dict])

    variance_df_str = pd.concat([overall_variances_df, variance_df_individual], ignore_index=True)
    
    for col in variance_df_str.columns:
        if col != 'FG/MFG (index min_Da)':
            # Apply the transformation: multiply by 100, keep only one decimal, 
            # convert to string, and then append '%'
            variance_df_str[col] = variance_df_str[col] * 100  # Multiply by 100
            variance_df_str[col] = variance_df_str[col].round(1)  # Keep only one decimal
            variance_df_str[col] = variance_df_str[col].astype(str) + '%'  # Convert to string and add '%'
    variance_df_str = pd.concat([overall_p_value_df, variance_df_str], ignore_index=True)
    
    return variance_df_individual, variance_df_str, levels_coef_determination_dict


# For meta analysis, including separated templates
def reassign_features(Featurelist, Rt_threshold, Mz_threshold):
    # Sort by feature number and reset index
    df = Featurelist.sort_values('feature').reset_index(drop=True)
    
    # Initialize a new column for the updated features
    df['new_feature'] = df['feature']
    
    for i in range(len(df)):
        # Get the current feature's rt, mz, and MS_level
        current_rt = df.at[i, 'rt']
        current_mz = df.at[i, 'm/z']
        current_feature = df.at[i, 'new_feature']
        current_ms_level = df.at[i, 'MS_level']
        
        # Define the rt and mz range to search for similar features
        rt_min = current_rt - Rt_threshold
        rt_max = current_rt + Rt_threshold
        mz_min = current_mz - Mz_threshold
        mz_max = current_mz + Mz_threshold
        
        # Find indices of matching features with the same MS_level
        matches = df[(df['rt'] >= rt_min) & (df['rt'] <= rt_max) & 
                     (df['m/z'] >= mz_min) & (df['m/z'] <= mz_max) &
                     (df['MS_level'] == current_ms_level) &
                     (df['new_feature'] != current_feature)].index
        
        # If matches are found, update their feature numbers
        if not matches.empty:
            min_feature = min(current_feature, df.at[matches[0], 'new_feature'])
            for match in matches:
                df.at[match, 'new_feature'] = min_feature
            df.at[i, 'new_feature'] = min_feature
    
    # Group by the new_feature and calculate the average rt and mz
    grouped = df.groupby('new_feature').apply(lambda x: pd.Series({
        'avg_rt': round(x['rt'].mean(), 2),
        'avg_mz': round(x['m/z'].mean(), 5)
    })).reset_index()
    
    # Merge the averages back into the original dataframe
    df = df.merge(grouped, on='new_feature', how='left')
    
    # Update the rt and mz values with the calculated averages
    df['rt'] = df['avg_rt']
    df['m/z'] = df['avg_mz']
    
    # Drop the temporary average columns
    df.drop(columns=['avg_rt', 'avg_mz'], inplace=True)
    
    # Remove duplicate samples within the same new_feature group
    df = df.drop_duplicates(subset=['new_feature', 'sample'])
    
    # Replace the original feature column with new_feature column
    df['feature'] = df['new_feature']
    df.drop(columns=['new_feature'], inplace=True)
    
    return df

def dropdowns(Sample_list):
    sample_list = Sample_list
    dropdown_items = [{"label": text, "value": text} for text in sample_list.index]
    dropdown_items_binary = [{"label": text, "value": text} for text in sample_list.index if len(sample_list.loc[text].unique()) == 2]  # dropdowns for labels which have binary condition and opposite condition as possibilities
    
    combinations = set()
    for non_binary_class in sample_list.index:
        conditions = list(sample_list.loc[non_binary_class].unique())
        if len(conditions) > 2:
            for i in range(len(conditions)):
                for j in range(i+1, len(conditions)):
                    combination1 = f"{conditions[i]} vs {conditions[j]}"
                    combination2 = f"{conditions[j]} vs {conditions[i]}"
                    if combination1 not in combinations and combination2 not in combinations:
                        combinations.add(f'{non_binary_class} - {combination1}')
    dropdown_items_binary.extend([{"label": text, "value": text} for text in combinations])
    dropdown_items_binary.sort(key=lambda item: item['label'])
    
    return dropdown_items, dropdown_items_binary
# Preprocess the data to facilitate comparison of features across samples
###############################################################################

# Set sd_table as a global variable, it is for standardized_table, meaning the featurelist which include standardized valeus of samples for each feature
all_final_mean = None
all_sample_mean = None
all_treatments = None
stat_result = None
clicked_node_identifier = None
current_dimension = '3D'
dropdown_items = [{"label": '', "value": ''}]
dropdown_items_binary = [{"label": '', "value": ''}]
featurelist_raw = None
featurelist = None
featurelist_raw = None
featurelistname = None
featurepath = None
FG_pair_shared_features = None
fg_table = None # ic table is the list of feature groups based on the defined feature network
fg_table_render =  pd.DataFrame()
G = nx.Graph() # Create the G network, meaning the python object for feature network
S = nx.Graph() # Network for the meta feature grouping
ion_df = None
ion_df_raw = None
shape_vector_store: Dict[str, np.ndarray] = {}
feature_shape_vectors: Dict[str, np.ndarray] = {}
shape_record_store: Dict[str, dict] = {}
SHAPE_VECTOR_POINTS = 80
levels = None
loading_progress = None
meta_intensities = None
minimum_missing_value_threshold = 0.5 # to test correlation in the network graph, two features needs to have a difference of minimum of 50% sample containing value to calculate the correlation. In case there are too many 0s. It depends on your treatment, theorically, if one treatment, one control it should be 0.5, but we can consider below if we have false negative in some samples
network_fig = None # Network graph figure
network_updated = False
potential_neutral_masses_groups = None
preprocessed_df = None
preprocessed_df_raw = None
pres_sample_threshold = None 
project_loaded = None
Project_loaded = False
project_initiated = False
stand_preprocessed_df = None
stand_preprocessed_df_raw = None
sample_list = None
sample_list_raw = None
sd_table = None
stand_ion_df = None
stat_proces_thread = None
treatment = None
treatment_selection = None
updating = False
validity = False
volcano_feature_groups = None


def _active_project():
    return project_loaded


def _active_project_name() -> Optional[str]:
    project = _active_project()
    if project is None:
        return None
    return getattr(project, 'project_name', getattr(project, 'name', None))


@contextmanager
def _feature_grouping_step_logger(step_name: str):
    project_name = _active_project_name()
    project_context = _active_project()
    logging_config.log_info(
        logger,
        "Feature grouping step started: %s",
        step_name,
        project=project_context,
    )
    try:
        yield
    except Exception as exc:  # pragma: no cover - defensive logging
        logging_config.log_exception(
            logger,
            "Feature grouping step failed: %s",
            step_name,
            project=project_context,
            exception=exc,
        )
        raise
    else:
        logging_config.log_info(
            logger,
            "Feature grouping step completed: %s",
            step_name,
            project=project_context,
        )


def _shape_payload_to_vector(shape_payload) -> Optional[np.ndarray]:
    if (
        not isinstance(shape_payload, (list, tuple))
        or len(shape_payload) != 2
    ):
        return None
    rts, intensities = shape_payload
    if not rts or not intensities:
        return None
    rts_array = np.asarray(rts, dtype=float)
    intensities_array = np.asarray(intensities, dtype=float)
    mask = np.isfinite(rts_array) & np.isfinite(intensities_array)
    if mask.sum() < 3:
        return None
    rts_array = rts_array[mask]
    intensities_array = intensities_array[mask]
    order = np.argsort(rts_array)
    rts_array = rts_array[order]
    intensities_array = intensities_array[order]
    apex_index = int(np.argmax(intensities_array))
    apex_rt = rts_array[apex_index]
    centered_rts = rts_array - apex_rt
    max_abs_rt = np.max(np.abs(centered_rts))
    if max_abs_rt > 0:
        normalized_rts = centered_rts / max_abs_rt
    else:
        normalized_rts = centered_rts
    intensities_mean = intensities_array.mean()
    intensities_std = intensities_array.std()
    if intensities_std > 0:
        normalized_intensities = (intensities_array - intensities_mean) / intensities_std
    else:
        normalized_intensities = intensities_array - intensities_mean
    grid = np.linspace(-1.0, 1.0, SHAPE_VECTOR_POINTS)
    vector = np.interp(
        grid,
        normalized_rts,
        normalized_intensities,
        left=0.0,
        right=0.0,
    )
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def _sanitize_shape_payload(shape_payload) -> Optional[tuple]:
    if (
        not isinstance(shape_payload, (list, tuple))
        or len(shape_payload) != 2
    ):
        return None
    rts, intensities = shape_payload
    if not rts or not intensities:
        return None
    rts_array = np.asarray(rts, dtype=float)
    intensities_array = np.asarray(intensities, dtype=float)
    mask = np.isfinite(rts_array) & np.isfinite(intensities_array)
    if mask.sum() < 2:
        return None
    rts_array = rts_array[mask]
    intensities_array = intensities_array[mask]
    order = np.argsort(rts_array)
    rts_array = rts_array[order]
    intensities_array = intensities_array[order]
    return rts_array.tolist(), intensities_array.tolist()


def _load_shape_record_store(store_path: Optional[str]) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    if not store_path or not os.path.isfile(store_path):
        return records
    try:
        with lz4.frame.open(store_path, mode="rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                shape_id = record.get("shape_id")
                cleaned_shape = _sanitize_shape_payload(record.get("shape"))
                if not shape_id or cleaned_shape is None:
                    continue
                record["shape"] = cleaned_shape
                records[str(shape_id)] = record
    except OSError:
        print("Unable to load chromatographic shapes; continuing without shape refinement.")
    return records


def _load_shape_vector_store(
    store_path: Optional[str],
    record_store: Optional[Dict[str, dict]] = None,
) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    source_records = None
    if record_store:
        source_records = record_store.items()
    elif store_path and os.path.isfile(store_path):
        try:
            parsed_records = []
            with lz4.frame.open(store_path, mode="rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed_records.append((record.get("shape_id"), record.get("shape")))
            source_records = parsed_records
        except OSError:
            print("Unable to load chromatographic shapes; continuing without shape refinement.")
            return vectors
    else:
        return vectors

    for shape_id, payload in source_records:
        if record_store is not None:
            vector = _shape_payload_to_vector(payload.get("shape"))
        else:
            vector = _shape_payload_to_vector(payload)
        if shape_id and vector is not None:
            vectors[str(shape_id)] = vector
    return vectors


def _build_feature_shape_vector_index(
    feature_df: Optional[pd.DataFrame],
    shape_vectors: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    feature_vectors: Dict[str, np.ndarray] = {}
    if (
        feature_df is None
        or shape_vectors is None
        or not shape_vectors
        or 'shape_id' not in feature_df.columns
    ):
        return feature_vectors
    mapping = (
        feature_df[['feature', 'shape_id']]
        .dropna(subset=['shape_id'])
        .copy(deep=True)
    )
    if mapping.empty:
        return feature_vectors
    mapping['shape_id'] = mapping['shape_id'].astype(str)
    for feature, group in mapping.groupby('feature'):
        vectors = [
            shape_vectors.get(shape_id)
            for shape_id in group['shape_id']
            if shape_id in shape_vectors
        ]
        vectors = [vec for vec in vectors if vec is not None]
        if not vectors:
            continue
        stacked = np.vstack(vectors)
        mean_vector = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
        feature_vectors[feature] = mean_vector
    return feature_vectors


def _initialize_shape_vectors(feature_df: Optional[pd.DataFrame]):
    global shape_vector_store, feature_shape_vectors, project_loaded, shape_record_store
    store_path = getattr(project_loaded, 'shape_store_path', None)
    shape_record_store = _load_shape_record_store(store_path)
    shape_vector_store = _load_shape_vector_store(store_path, shape_record_store)
    feature_shape_vectors = _build_feature_shape_vector_index(
        feature_df,
        shape_vector_store,
    )


def _refresh_feature_shape_vectors(feature_df: Optional[pd.DataFrame]):
    global feature_shape_vectors, shape_vector_store
    if not shape_vector_store:
        feature_shape_vectors = {}
        return
    feature_shape_vectors = _build_feature_shape_vector_index(
        feature_df,
        shape_vector_store,
    )

def initiate_project(Project):
    global featurelist, featurelist_raw, featurepath, featurelistname
    global treatment, project_loaded
    global preprocessed_df, preprocessed_df_raw, stand_preprocessed_df, stand_preprocessed_df_raw
    global ion_df, ion_df_raw, stand_ion_df
    global levels, sample_list, sample_list_raw, all_treatments, treatment_selection
    global dropdown_items, dropdown_items_binary
    global fg_table, fg_table_render
    global shape_vector_store, feature_shape_vectors
    
    project_loaded = Project
    featurelist = project_loaded.msn_df_deblanked
    featurelist = featurelist.drop([col for col in featurelist.columns if not col or col.strip() == ''], axis=1)
    featurelist = featurelist.drop(columns = ['label'])
    treatment = project_loaded.treatment
    fg_table, fg_table_render = None, pd.DataFrame() 
    

    # Find the list of sampels (without blank)
    classes = []
    for t, c in treatment.keys():
        if c not in classes:
            classes.append(c) # take list of classes
    first_class = classes[0] # take ont class, the first one for example, to at the end behing able to retrieve all sampels, because in each class all samples are represetented, segrouped by treatment
    samples = [] # list of samples
    for k, v in treatment.items():
        if k[1] == first_class: #k[1] is the class
            samples.extend(v)
            
    featurelist = add_zeros(featurelist, samples) # elongate the featurelist adding the samples with 0 values, which have been removed during the untargeted pipeline for simplicity
    
    featurelist_raw = featurelist.copy(deep=True) # back up of the variable, necessary for the filtering option tool
    featurepath = project_loaded.featurepath
    featurelistname = project_loaded.name
    _initialize_shape_vectors(featurelist)
    

    # Add the rt and m/z values for each feature
    rt_mz_df = featurelist[['feature', 'rt', 'm/z']].drop_duplicates().set_index('feature')
    absolute_preprocessed_df = featurelist.pivot_table(index='feature', columns='sample', values='area', fill_value=0).merge(rt_mz_df, left_index=True, right_index=True)
    
    # Relative abundance normalization
    stand_preprocessed_df = absolute_preprocessed_df.drop(columns = ['rt', 'm/z'])
    stand_preprocessed_df = stand_preprocessed_df.div(stand_preprocessed_df.max(axis=1), axis=0)
    stand_preprocessed_df['rt'] = absolute_preprocessed_df['rt']
    stand_preprocessed_df['m/z'] = absolute_preprocessed_df['m/z']
    
    stand_preprocessed_df_raw = stand_preprocessed_df.copy(deep=True)
    preprocessed_df = absolute_preprocessed_df
    preprocessed_df_raw = absolute_preprocessed_df.copy(deep=True) # back up of the variable, necessary for the filtering option tool
    
    # Finally, we have ion_df, each row represent a feature (feature), all samples are in columns, values are height or area, there is a also rt col, m/z co and ANN col for the quality score per feature
    # ion_df = preprocessed_df.copy(deep=True) # ion_df is used to make PCA, volcano, pld-da
    # ion_df_raw = ion_df.copy(deep=True) # back up of the variable, necessary for the filtering option tool
    # stand_ion_df = ion_df.copy(deep=True)
    
    # Take treatment level for each samples which are not pool or std (blank already removed)
    # The unique treatment are removed now. Must not be removed durign the pipeline in case there are multiple unique treatments acrros the exepreiments in a metaanalysis, typically the date
    all_classes_lenght = {} # take into a dict all different classes, and as values the number of different treatment for each class,. The idea is to identify the classes with only one treatment, to remove them
    for t, c in treatment.keys():
        if c not in all_classes_lenght:
            all_classes_lenght[c] = 1
        else:
            all_classes_lenght[c] += 1
    keys_with_value_one = [k for k, v in all_classes_lenght.items() if v == 1]
    keys_with_value_one = f'|{"|".join(keys_with_value_one)}'
    filt_treatment = {k: v for k, v in treatment.items() if not re.search(f'Pool|Std{keys_with_value_one}', k[1])}
    levels = [] # correspond to the level treatment class, like 'bx', 'technician', 'project', ...
    for i in filt_treatment.keys():
        if i[1] not in levels:            
            levels.append(i[1])
            
    # Create sample_list df, wehre levels are indexes, samplesa re rows, values of samples are corresponding labels for each level        
    sample_list = {s:levels for s in samples}        
    sample_list = pd.DataFrame(sample_list, index=levels) # df number of rows is equal to levels lenght, meaning number of treaetment classes
    sample_list.index.name = 'levels'
    for s in sample_list:
        for k,v in filt_treatment.items(): # k is a tuple: (treatment name, treatment class)
            if s in v:
                sample_list.loc[k[1], s] = k[0]
    sample_list_raw = sample_list.copy(deep=True) # back up of the variable, necessary for the filtering option tool
    
    # Take all thre treatment, no matter the level, arranged as dict for checkboxes
    all_treatments = []
    for ix, row in sample_list.iterrows():
        for treat in row.unique():
            all_treatments.append({"label": f'{treat} ({ix})', "value": treat})
    treatment_selection = [t['value'] for t in all_treatments]

    # Generate Dropdown Menu Items based on the Featurelist's index
    dropdown_items, dropdown_items_binary  = dropdowns(sample_list)

    print('project initiated')

# Meta Feature Group part
###############################################################################

# Function to verify if two FG are from the same experiment
def check_FG(Experiment_FG, FG1, FG_grouping):
    exp1 = Experiment_FG[FG1]['experiment']
    FG_grouping = FG_grouping.values[0] # take the list from the Serie
    for f in FG_grouping:        
        exp2 = Experiment_FG[f]['experiment']
        if FG1 == f:
            return True # False indicate a valid association, intra FG is valid, inter FG inside same exp is not valid, inter FG between exp is valid
        elif exp1 == exp2: # FG from same experiment but not same FG, hiwhc is checked above
            return False
    return True

# Function to check if a component can accept a new edge while respecting the constraints
def can_add_edge(fg1, fg2, edge_nature, component, component_edges, experiment_index):
    if edge_nature in component_edges:
        return False
    if len(component) >= len(experiment_index):
        return False
    return True

# Function to find or create a component for an edge
def find_or_create_component(fg1, fg2, edge_nature, components, component_edges, components_weights, experiment_index):
    for idx, component in enumerate(components):
        if (fg1 in component or fg2 in component) and can_add_edge(fg1, fg2, edge_nature, component, component_edges[idx], experiment_index):
            return idx
    # If no suitable component is found, create a new one
    components.append(set())
    component_edges.append(set())
    components_weights.append(0)
    return len(components) - 1


def initiate_project_meta(Project):
    global featurelist, featurelist_raw, featurepath, featurelistname
    global treatment, project_loaded
    global preprocessed_df, preprocessed_df_raw, stand_preprocessed_df, stand_preprocessed_df_raw
    global ion_df, ion_df_raw, stand_ion_df
    global levels, sample_list, sample_list_raw, all_treatments, treatment_selection
    global dropdown_items, dropdown_items_binary
    global S, FG_pair_shared_features, potential_neutral_masses_groups
    global fg_table, fg_table_render
    global shape_vector_store, feature_shape_vectors
    
    project_loaded = Project
    
    # Treatment labelling part
    ##########################
    featurepath = project_loaded.featurepath
    featurelistname = project_loaded.name
    featurelist = project_loaded.msn_df_deblanked
    featurelist = featurelist.drop([col for col in featurelist.columns if not col or col.strip() == ''], axis=1)
    featurelist = featurelist.drop(columns = ['label'])
    featurelist_raw = featurelist.copy(deep=True)
    _initialize_shape_vectors(featurelist)
    treatment = project_loaded.treatment
    fg_table, fg_table_render = None, pd.DataFrame() 
    # Find the list of sampels (without blank)
    classes = []
    for t, c in treatment.keys():
        if c not in classes:
            classes.append(c) # take list of classes
    first_class = classes[0] # take ont class, the first one for example, to at the end behing able to retrieve all samples, because in each class all samples are represetented, segrouped by treatment
    samples = [] # list of samples
    for k, v in treatment.items():
        if k[1] == first_class: #k[1] is the class
            samples.extend(v)

    # Take treatment level for each samples which are not pool or std (blank already removed)
    # The unique treatment are removed now. Must not be removed durign the pipeline in case there are multiple unique treatments acrros the exepreiments in a metaanalysis, typically the date
    all_classes_lenght = {} # take into a dict all different classes, and as values the number of different treatment for each class,. The idea is to identify the classes with only one treatment, to remove them
    for t, c in treatment.keys():
        if c not in all_classes_lenght:
            all_classes_lenght[c] = 1
        else:
            all_classes_lenght[c] += 1
    keys_with_value_one = [k for k, v in all_classes_lenght.items() if v == 1]
    keys_with_value_one = f'|{"|".join(keys_with_value_one)}'
    filt_treatment = {k: v for k, v in treatment.items() if not re.search(f'Pool|Std{keys_with_value_one}', k[1])}
    levels = [] # correspond to the level treatment class, like 'bx', 'technician', 'project', ...
    for i in filt_treatment.keys():
        if i[1] not in levels:            
            levels.append(i[1])
            
    # Create sample_list df, wehre levels are indexes, samplesa re rows, values of samples are corresponding labels for each level        
    sample_list = {s:levels for s in samples}        
    sample_list = pd.DataFrame(sample_list, index=levels) # df number of rows is equal to levels lenght, meaning number of treaetment classes
    sample_list.index.name = 'levels'
    for s in sample_list:
        for k,v in filt_treatment.items(): # k is a tuple: (treatment name, treatment class)
            if s in v:
                sample_list.loc[k[1], s] = k[0]
    sample_list_raw = sample_list.copy(deep=True) # back up of the variable, necessary for the filtering option tool
    
    # Take all the treatment, no matter the level, arranged as dict for checkboxes
    all_treatments = []
    for ix, row in sample_list.iterrows():
        for treat in row.unique():
            all_treatments.append({"label": f'{treat} ({ix})', "value": treat})
    treatment_selection = [t['value'] for t in all_treatments]

    # Generate Dropdown Menu Items based on the Featurelist's index
    dropdown_items, dropdown_items_binary  = dropdowns(sample_list)

    print('meta project initiated')

# Define the layout of the app
###############################################################################

# Define the PCA, volcano, and plsda layout tabs separately for clarity
pca_tab_content = dbc.Card(
    dbc.CardBody(
        dbc.Row([
            dbc.Col(html.Div([dbc.Spinner(dcc.Graph(id='pca-plot', style={"height": "400px"}), color="light", spinner_style={"width": "3rem", "height": "3rem"}),
                              dbc.Button("2D", id="dimension-button", color="primary", style={
                                 "position": "absolute",
                                  "top": "5px",
                                  "left": "10px",
                                  "zIndex": 1000,
                                  "width": "50px",
                                  "height": "40px",
                                  "border-radius": "50%",
                                  "padding": "0",
                                  "text-align": "center",
                                  "line-height": "40px"}),
                                  html.Div(id='intermediate-signal-pca', style={'display': 'none'})
                                  ]),  # Hidden Div
                    width=8),  
            
            dbc.Col([
                html.P("PCA", className="card-text"),
                dbc.InputGroup(
                    [dbc.InputGroupText("Title"), dbc.Input(id='pca-title', placeholder="Enter a title here...")], size="sm",
                    className="mb-2",
                    ),
                
                html.Div([
                    dbc.Label("Data nature"),
                    dbc.RadioItems(
                        options=[{"label": "Raw feature", "value": "f"},
                                 {"label": "Standardized feature", "value": "fz"}],
                                 
                        value="fz",
                        id="pca-nature",
                        inline=True,
                    ),
                    dbc.Tooltip(
                        "Define the nature of the feature values. Raw feature are the direct values from the feature list, area under the curve. Standardized is standardization across samples per unique feature.",
                        target="pca-nature",  # ID of the component to which the tooltip is attached
                        placement="left",
                    ),
                    ], style={'marginBottom': '10px'}),
                
                html.Div([                        
                        dbc.Label("Labelling option"),
                        dbc.RadioItems(
                            options=dropdown_items,
                            value=dropdown_items[0]['value'],
                            id="pca-level",
                            inline=True,
                        ),
                        dbc.Tooltip(
                            children="Define how to label your data points based on a treatment level or even metadata",
                            target="pca-level",  # ID of the component to which the tooltip is attached
                            placement="left",
                            style={'width': '200px'}
                        ),
                        ], style={'marginBottom': '10px'}),
                
                html.Div([
                        dbc.Label("Type of data point"),
                        dbc.RadioItems(
                            options=[
                                {"label": "Samples", "value": 1},
                                {"label": "Feature groups", "value": 2},
                            ],
                            value=1,
                            id="pca-type"),
                        ]),              
                dbc.Tooltip(
                    "Define if you display samples as data points or grouped features",
                    target="pca-type",  # ID of the component to which the tooltip is attached
                    placement="right"), 
                ]), 
                ])
            ), className="mt-1"
)

plsda_tab_content = dbc.Card(
    dbc.CardBody(
        dbc.Row([
            dbc.Col(html.Div([dbc.Spinner(dcc.Graph(id='plsda-plot', style={"height": "400px"}), color="light", spinner_style={"width": "3rem", "height": "3rem"}),
                    html.Div(id='intermediate-signal-plsda', style={'display': 'none'})]),  # Hidden Div
                    width=7),
            
            dbc.Col([
                html.P("PLS-DA", className="card-text"),
                
                dbc.InputGroup(
                [dbc.InputGroupText("Title"), dbc.Input(id='plsda-title', placeholder="Enter a title here...")], size="sm",
                className="mb-2"),
                
                html.Div([
                    dbc.Label("Data nature"),
                    dbc.RadioItems(
                        options=[{"label": "Raw feature", "value": "f"},
                                 {"label": "Standardized feature", "value": "fz"}],
                                 
                        value="fz",
                        id="plsda-nature",
                        inline=True,
                    ),
                    dbc.Tooltip(
                        "Define the nature of the feature values. Standardized is standardization across samples per unique feature.",
                        target="pca-nature",  # ID of the component to which the tooltip is attached
                        placement="left",
                    ),
                    ], style={'marginBottom': '10px'}),
                html.Div([
                        dbc.Label("Labelling option"),
                        dbc.RadioItems(
                            options=dropdown_items,
                            value=dropdown_items[0]['value'],
                            id="plsda-level",
                            inline=True),
                        
                        dbc.Tooltip(
                            "Define how to label your data points based on a treatment level or even metadata",
                            target="plsda-level",  # ID of the component to which the tooltip is attached
                            placement="left"),
                        
                        ], style={'marginBottom': '10px'}),
  
            ]),
        ])
    ), className="mt-1"
)

volcano_tab_content = dbc.Card(
                        dbc.CardBody([
                            html.Div(id='volcano-table-progress-div', 
                                     children=[html.P("Processing the statistical tests of the feature groups"),
                                               html.Br(),
                                               dbc.Progress(id = 'volcano-table-progress', value=0, striped=True)], style={'height': '80px'}),
                            html.Div(id='volcano-table-div',
                            children=[dbc.Row([
                                dbc.Col(
                                    html.Div([
                                        dbc.Spinner(dcc.Graph(id='volcano-plot', style={"height": "400px"}), color="light", spinner_style={"width": "3rem", "height": "3rem"}),
                                        dbc.Button("Sig", id="all-significant", color="primary", style={
                                           "position": "absolute",
                                            "top": "5px",
                                            "left": "10px",
                                            "zIndex": 1000,
                                            "width": "50px",
                                            "height": "40px",
                                            "border-radius": "50%",
                                            "padding": "0",
                                            "text-align": "center",
                                            "line-height": "40px"}),
                                       
                                        html.Div(id='intermediate-signal-volcano', style={'display': 'none'})
                                                ], style={"position": "relative"}),  # Hidden Div
                                        width=7),
                                dbc.Col([
                                    html.P("Volcano Plot", className="card-text"),
                                    dbc.InputGroup(
                                    [dbc.InputGroupText("Title"), dbc.Input(id='volcano-title', placeholder="Enter a title here...")], size="sm",
                                    className="mb-2",
                                    ),
                                    
                                    html.Div([
                                            dbc.Label("Labelling option", id="volcano-level-label"),
                                            # html.Div([
                                                dbc.Select(
                                                    options=dropdown_items_binary,
                                                    value=dropdown_items_binary[0]['value'],
                                                    id="volcano-level",
                                                ),
                                            # ], style={'max-height': '120px', 'overflow-y': 'auto'}),  # Adjust height as needed based on your line height
                                           
                                            dbc.Tooltip(
                                                "Define how to label your data points based on a treatment level or even metadata",
                                                target="volcano-level-label",  # ID of the component to which the tooltip is attached
                                                placement="left"),
                                            
                                            ], style={'marginBottom': '10px'}),
                                    
                                    html.Div([
                                            dbc.Label("Significance level"),                        
                                            dbc.RadioItems(
                                                options=[
                                                    {"label": "0.05", "value": 0.05},
                                                    {"label": "0.01", "value": 0.01},
                                                    {"label": "0.001", "value": 0.001},
                                                    {"label": "0.0001", "value": 0.0001}],
                                                value= 0.05,
                                                id="volcano-pvalue",
                                                inline=True),
                                            ], style={'marginBottom': '10px'}),  
                                    html.Div([
                                            dbc.Label("Chi square"),                        
                                            html.Div(
                                                id='chi-square-res',
                                                children='',
                                                style={
                                                    
                                                    'backgroundColor': 'rgba(211, 211, 211, 0.7)',  # Light grey with 50% transparency
                                                    'padding': '10px',  # Add some padding if needed
                                                    'borderRadius': '5px',  # Optional: Add rounded corners
                                                    'margin': '0',  # Remove any margin if needed
                                                    'margin-right': '5px',  # Set the right margin to 5px
                                                }),
                                            ], style={'marginBottom': '10px'}),  
                                        ]),
                                    ])
                                ], style={'display':'none'}),
                        ]), className="mt-1"
                    )
                                    
fg_table_content =    dbc.Card(                                    
                            dbc.CardBody([
                                html.Div(id='fg-table-progress-div', 
                                         children=[html.P("Processing the statistical tests of the feature groups"),
                                                   html.Br(),
                                                   dbc.Progress(id = 'fg-table-progress', value=0, striped=True)], style={'height': '80px'}),
                                html.Div(id='fg-table-div',
                                    children=[
                                    html.Div([dash_table.DataTable(
                                        id='fg-table',
                                        columns = [],
                                        data = [],
                                        sort_action="native",
                                        page_size=10,
                                        row_selectable="single",
                                        cell_selectable=False,
                                        sort_by=[{'column_id': 'Size', 'direction': 'desc'}],
                                        style_table={
                                            'backgroundColor': 'rgb(250, 250, 250)',
                                            'overflowX': 'auto'},  # Example: light gray table background
                                        style_header={
                                            'fontFamily': '"Helvetica Neue", Arial, sans-serif',
                                            'backgroundColor': 'rgb(235, 235, 250)',
                                            'color': 'rgb(0, 0, 0)',
                                            'fontWeight': 'bold'},
                                        style_cell={
                                            'fontFamily': '"Helvetica Neue", Arial, sans-serif',
                                            'backgroundColor':'rgb(250, 250, 250)',  # Example: dark blue cell background
                                            'color': 'rgb(0, 0, 0)',            # Example: white text color
                                            'textAlign': 'left',
                                            'overflow': 'hidden',
                                            'fontSize': '12px',
                                            'textOverflow': 'ellipsis',
                                            'padding': '5px'},
                                        style_data={
                                            'borderColor': 'rgb(102, 102, 102)',  # Example: black border for data cells
                                            'borderWidth': '1px',
                                            'borderStyle': 'solid'},
                                        style_data_conditional=[{
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(245, 245, 245)' }],
                                                        ),
                                            ]),  # Placeholder for the DataTable
                                        html.Div(
                                            [html.P([html.B("FG"), ": feature group index  /  ", html.B("Size:"), " number of features in the feature group  /  ", ], style={'height': '0px'}),
                                            html.P([html.B("cor_p:"), "Benjamini-Holchberg corrected p-value  /  ", html.B("p:"), " p-value of the test  /  ", html.B("np-p:"), " non-parametric p-value"], style={'height': '0px'}),
                                            html.P([html.B("np-cor-p:"), "Benjamini-Holchberg corrected non-parametric p-value  /  ", html.B("a:"), " assumptions validity of the test"], style={'height': '0px'}),
                                            html.P([html.B(" -:"), " not enough data"], style={'height': '0px'})],
                                            style={
                                                'width': '650px',
                                                'height': '65px',
                                                'color': '#e3e3e3',  # or any color you want
                                                'top':'5px',
                                                'fontFamily': 'Arial',
                                                'fontSize': '12px',
                                                'zIndex': '1000',  # ensures the text is on top
                                                'overflow': 'visible'
                                                }),
                                        ], style={'display':'none'}),
                                    ]), className="mt-1"
                                )


# Below is the main layout, which include the tabs        
main_layout_deprecated =  html.Div([
                    html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
                    html.Div([dbc.Button("Display analytical dashboard", id='first-update-button', n_clicks=0, color="primary"),
                              html.Div(id='info-update-button'),
                                          ], 
                            style={'position': 'absolute',  # Positioning the div absolutely inside its relative parent
                                    'top': '50%',            # Centering vertically
                                    'left': '50%',           # Centering horizontally
                                    'transform': 'translate(-50%, -50%)'  # Adjusting the exact center position
                                }, id = "first-button-display"),
                    
                    html.Div([
                    # Wrap the Graph in a Loading component
                    dcc.Interval(id='update-interval', interval=500, n_intervals=0, max_intervals=-1, disabled = True), 
                    dcc.Interval(id='network-interval', interval=1000, n_intervals=0, max_intervals=-1), 
                    html.Div(id='network-intermediate', style={'display': 'none'}),
                    dbc.Spinner(dcc.Graph(figure={'data': []}, id='network-graph', style={"height": "350px", "margin-bottom":"10px"}), 
                                color="light", spinner_style={"width": "3rem", "height": "3rem"}),
                  
                   html.Div([
                        dbc.InputGroup(
                            [dbc.InputGroupText("RT +-", id='rt-input'),
                                dbc.Input(id='rt-threshold', type="number", value=0.02, step=0.01, min=0, max=10),
                                dbc.InputGroupText("min")],
                                className="me-1",
                                size="sm"), 

                        dbc.InputGroup(
                            [dbc.InputGroupText("Spearman Corr.", id='spearman-input'),
                             dbc.Input(id='correlation-threshold',
                                       type="number", value=0.9, step=0.01, min=0, max=1)],
                            size="sm"
                                ),
                        dbc.Tooltip(
                            "RT threshold to create a FG or a MFG.", #  It need to have at least 50% of same samples with values != 0 in both features to be valid. I.e you have 10 samples in total, in f1 if there is 10 samples with values != 0, and in f2 at least 5 samples with values !=0, the correlation is calculated.
                            target="rt-threshold",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        dbc.Tooltip(
                            "NORMAL MODE: Spearman correlation accross samples for each pair of feature. META MODE: m/z threshold between two potential neutral mass.", #  It need to have at least 50% of same samples with values != 0 in both features to be valid. I.e you have 10 samples in total, in f1 if there is 10 samples with values != 0, and in f2 at least 5 samples with values !=0, the correlation is calculated.
                            target="spearman-input",  # ID of the component to which the tooltip is attached
                            placement="top"),
                    html.Div([
                        dbc.Container([
                            dbc.ButtonGroup([
                                dbc.Button("Update", id='update-button', n_clicks=0, color="primary"),
                                dbc.Button("Filtering", id='filter-button', n_clicks=0, color="secondary"),
                                            ], class_name='equal-width-buttons'),
                            dbc.ButtonGroup([
                                dbc.Button("Export RDA", id='export-button', n_clicks=0, color="info"),
                                dbc.Button("Variation", id='variation-button', n_clicks=0, color="warning"),
                                ], class_name='equal-width-buttons'),
                                ],
                                    style={'text-align': 'center'},  # Center-align the content inside the container
                                    fluid=True, className="d-flex flex-column justify-content-between"),
                        dbc.Container([                            
                            dbc.InputGroup(
                                [dbc.InputGroupText("Sample threshold"), 
                                 dbc.Input(id='sample-threshold', type="number", value=100, step=1, min=0, max=100),
                                 dbc.InputGroupText("%")]
                                    ),
                            dbc.Button("(Meta) Update Intensities", id='update-intensities', n_clicks=0, color="primary"),],
                                style={'text-align': 'center'},  # Center-align the content inside the container
                                fluid=True, className="d-flex flex-column justify-content-between"),
                        dbc.Tooltip(
                            "In a Meta analysis, it calculates the statistics based on raw intensity value for each sample, not on the presence/absence.",
                            target="update-intensities",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        dbc.Tooltip(
                        "Realize a statiscal test only if at least a signal is found in threshold % of the samples for each condition in a treatment level.",
                            target="sample-threshold",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        dbc.Tooltip(
                            "Display the variance explained per feature to help identify discriminatory groups across treatments.",
                                target="variation-button",  # ID of the component to which the tooltip is attached
                                placement="bottom"),
                            ], style={'display': 'flex','margin-bottom': '10px'}),
                    ], style={'display': 'inline-block', 'width': '35%', 'margin-left': '10px',  'margin-right': '10px', 'margin-top': '10px'}),
                html.Div([
                    dbc.Tabs(
                        [
                            dbc.Tab(pca_tab_content, label=" PCA "),
                            dbc.Tab(plsda_tab_content, label=" PLS-DA "),
                            dbc.Tab(volcano_tab_content, label=" Volcano plot "),
                            dbc.Tab(fg_table_content, label=" Feature group list "),
                        ]
                            )
                        ], style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-left': '10px', 'margin-top': '10px'}),
            
                html.Div([
                       html.Hr(style={'borderTop': '2px powdergrey', 'margin': '5px 0'  })
                         ], style={'padding': '5px'}),
                html.Div(id='dynamic-content', style={'margin-left': '10px',  'margin-right': '10px'}),
                ], id='main-content', style = {'display':'none', 'margin-top': '50px'})
                ])

# Below is the main layout, which include the tabs        
main_layout =  html.Div([
                    html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
                    html.Div([dbc.Button("Display analytical dashboard", id='first-update-button', n_clicks=0, color="primary"),
                              html.Div(id='info-update-button'),
                                          ], 
                            style={'position': 'absolute',  # Positioning the div absolutely inside its relative parent
                                    'top': '50%',            # Centering vertically
                                    'left': '50%',           # Centering horizontally
                                    'transform': 'translate(-50%, -50%)'  # Adjusting the exact center position
                                }, id = "first-button-display"),
                    
                    html.Div([
                    html.Div([
                    # Wrap the Graph in a Loading component
                    dcc.Interval(id='update-interval', interval=500, n_intervals=0, max_intervals=-1, disabled = True), 
                    dcc.Interval(id='network-interval', interval=1000, n_intervals=0, max_intervals=-1), 
                    html.Div(id='network-intermediate', style={'display': 'none'}),
                    dbc.Spinner(dcc.Graph(figure={'data': []}, id='network-graph', style={"height": "350px", "margin-bottom":"10px"}), 
                                color="light", spinner_style={"width": "3rem", "height": "3rem"}),
                  
                   html.Div([
                        dbc.InputGroup(
                            [dbc.InputGroupText("RT +-", id='rt-input'),
                                dbc.Input(id='rt-threshold', type="number", value=0.02, step=0.01, min=0, max=10),
                                dbc.InputGroupText("min")],
                                className="me-1",
                                size="sm"), 

                        dbc.InputGroup(
                            [dbc.InputGroupText("Spearman Corr.", id='spearman-input'), 
                             dbc.Input(id='correlation-threshold', 
                                       type="number", value=0.9, step=0.01, min=0, max=1)],
                            size="sm"
                                ),
                        dbc.Tooltip(
                            "RT threshold to create a FG or a MFG.", #  It need to have at least 50% of same samples with values != 0 in both features to be valid. I.e you have 10 samples in total, in f1 if there is 10 samples with values != 0, and in f2 at least 5 samples with values !=0, the correlation is calculated.
                            target="rt-threshold",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        dbc.Tooltip(
                            "NORMAL MODE: Spearman correlation accross samples for each pair of feature. META MODE: m/z threshold between two potential neutral mass.", #  It need to have at least 50% of same samples with values != 0 in both features to be valid. I.e you have 10 samples in total, in f1 if there is 10 samples with values != 0, and in f2 at least 5 samples with values !=0, the correlation is calculated.
                            target="spearman-input",  # ID of the component to which the tooltip is attached
                            placement="top")], style={'display': 'flex','margin-bottom': '10px'}),
                    html.Div([
                        dbc.Container([
                            dbc.ButtonGroup([
                                dbc.Button("Update", id='update-button', n_clicks=0, color="primary"),
                                dbc.Button("Filtering", id='filter-button', n_clicks=0, color="secondary"),
                                            ], class_name='equal-width-buttons'),
                            dbc.ButtonGroup([
                                dbc.Button("Export RDA", id='export-button', n_clicks=0, color="info"),
                                dbc.Button("Variation", id='variation-button', n_clicks=0, color="warning"),
                                ], class_name='equal-width-buttons'),
                                ],
                                    style={'text-align': 'center'},  # Center-align the content inside the container
                                    fluid=True, className="d-flex flex-column justify-content-between"),
                        dbc.Container([                            
                            dbc.InputGroup(
                                [dbc.InputGroupText("Sample threshold"), 
                                 dbc.Input(id='sample-threshold', type="number", value=100, step=1, min=0, max=100),
                                 dbc.InputGroupText("%")]
                                    ),
                            dbc.Button("(Meta) Update Intensities", id='update-intensities', n_clicks=0, color="primary"),],
                                style={'text-align': 'center'},  # Center-align the content inside the container
                                fluid=True, className="d-flex flex-column justify-content-between"),
                        dbc.Tooltip(
                            "In a Meta analysis, it calculates the statistics based on raw intensity value for each sample, not on the presence/absence.",
                            target="update-intensities",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        dbc.Tooltip(
                        "Display the variance explained per feature to help identify discriminatory groups across treatments.",
                            target="variation-button",  # ID of the component to which the tooltip is attached
                            placement="bottom"),
                        dbc.Tooltip(
                        "Realize a statiscal test only if at least a signal is found in threshold % of the samples for each condition in a treatment level.",
                            target="sample-threshold",  # ID of the component to which the tooltip is attached
                            placement="top"),
                            ], style={'display': 'flex','margin-bottom': '10px'}),
                    ], style={'display': 'inline-block', 'width': '35%', 'margin-left': '10px',  'margin-right': '10px', 'margin-top': '10px'}),
                html.Div([
                    dbc.Tabs(
                        [
                            dbc.Tab(pca_tab_content, label=" PCA "),
                            dbc.Tab(plsda_tab_content, label=" PLS-DA "),
                            dbc.Tab(volcano_tab_content, label=" Volcano plot "),
                            dbc.Tab(fg_table_content, label=" Feature group list "),
                        ]
                            )
                        ], style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-left': '10px', 'margin-top': '10px'}),
            
                html.Div([
                       html.Hr(style={'borderTop': '2px powdergrey', 'margin': '5px 0'  })
                         ], style={'padding': '5px'}),
                html.Div(id='dynamic-content', style={'margin-left': '10px',  'margin-right': '10px'}),
                ], id='main-content', style = {'display':'none', 'margin-top': '50px'})
                ])

# Main layout callback
@callback(
    [Output('first-button-display', 'style'),
     Output('main-content', 'style'),
     Output('update-intensities', 'disabled'),
     Output('spearman-input', 'children'),
     Output('correlation-threshold', 'value'),
     Output('correlation-threshold', 'step'),
     Output('rt-input', 'children'),
     Output('rt-threshold', 'step'),
     Output("analytics-project-menu", 'children'),
     Output('filter-button', 'disabled')],
    Input('first-update-button', 'n_clicks'),
    prevent_initial_call = True
)
def update_main_layout(n_clicks):
    global validity, Project_loaded
    Project_loaded = cache.get('project_loaded')
    validity = False
    if Project_loaded:
        try:
            if Project_loaded.treatment != None:
                project_name = Project_loaded.name
                print(project_name)
                if Project_loaded.meta:
                    disabled_state = False
                    initiate_project_meta(Project_loaded)
                    spearman_input_text = 'MGF m/z +-'
                    mz_threshold_value = 0.001
                    mz_threshold_step = 0.0001
                    rt_input_text = 'MGF RT +-'
                    rt_threshold_step = 0.001
                    disabled = True
                else:
                    disabled_state = True
                    spearman_input_text = dash.no_update
                    mz_threshold_value = dash.no_update
                    mz_threshold_step = dash.no_update
                    rt_threshold_step = dash.no_update
                    rt_input_text = dash.no_update
                    initiate_project(Project_loaded)
                    disabled = None
                validity = True
                return {'display':'none'}, {'margin-top': '50px'}, disabled_state, spearman_input_text, mz_threshold_value, mz_threshold_step, rt_input_text, rt_threshold_step, project_name, disabled
        except Exception:
            validity = 'Error'
            Project_loaded = False
    else:
        return {'position': 'absolute',  # Positioning the div absolutely inside its relative parent
                'top': '50%',            # Centering vertically
                'left': '50%',           # Centering horizontally
                'transform': 'translate(-50%, -50%)'  # Adjusting the exact center position
            }, {'display':'none', 'margin-top': '50px'}, True, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, '', True
    raise PreventUpdate()
    
@callback(
    [Output('first-update-button', 'children'),
     Output('info-update-button', 'children'),
     Output('update-interval', 'disabled'),
     ],
    [Input('first-update-button', 'n_clicks'),
     Input('update-interval', 'n_intervals')],
     prevent_initial_call = True
    )
def spinner_update_button(n_clicks, n):
    global validity, Project_loaded
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'first-update-button':
        return dbc.Spinner(color="light", size="sm"), 'Loading', False
        
    if button_id == 'update-interval':
        if validity == True:
            return dash.no_update, '',  True
        elif validity == False:
            if Project_loaded == None:
                Project_loaded = False
                
                return "Display analytical dashboard", 'No project loaded!', True
            elif Project_loaded == False:
                return dash.no_update, 'Loading', False
        elif validity == 'Error':
            return 'Failure', 'Your project contains error(s), you must delete it and restart the pipeline.', dash.no_update
    
    raise PreventUpdate()

# Callback to trigger after 'network-graph' is rendered. Indeed, we want PCA, volcano and plsda to be updated once the first default network graph is generated
@callback(
    Output('intermediate-signal-pca', 'children'),
    Input('network-graph', 'figure')
)
def update_intermediate_signal_pca(network_graph_figure):
    if network_updated == False:
        return dash.no_update
    return "network-graph-rendered" # this text is not display because in the corresponding style in the layout it is none.
    
@callback(
    Output('intermediate-signal-volcano', 'children'),
    Input('network-graph', 'figure')
)
def update_intermediate_signal_volcano(network_graph_figure):
    if network_updated == False:
        return dash.no_update
    return "network-graph-rendered" # this text is not display because in the corresponding style in the layout it is none.
   
@callback(
    Output('intermediate-signal-plsda', 'children'),
    Input('network-graph', 'figure')
)
def update_intermediate_signal_plsda(network_graph_figure):
    if network_updated == False:
        return dash.no_update
    return "network-graph-rendered" # this text is not display because in the corresponding style in the layout it is none.
   

# Callback for the dynamic content. This content is composed of histogram and feature table for a given ion (a "precusor" ion group is composed of one or multiple "fragment" features), and also filtering options
@callback(
    Output('dynamic-content', 'children'),
    [Input('network-graph', 'clickData'),
     Input('volcano-plot', 'clickData'),
     Input('filter-button', 'n_clicks'),
     Input('pca-plot', 'clickData'),
     Input('fg-table', 'selected_rows'),
     Input('variation-button', 'n_clicks'),
     Input('export-button', 'n_clicks'),
     Input('all-significant', 'n_clicks')],
    prevent_initial_call=True
)
def display_dynamic_content(network_clickData, volcano_clickData, filter_button_n_clicks, pca_clickData, fg_table_selected_row, variation_clicks, export_clicks, sig_clicks):    
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'network-graph':
        sd_table_content  = display_hist_n_table(network_clickData)
        return sd_table_content
    
    elif triggered_id == 'volcano-plot':        
        sd_table_content  = display_hist_n_table(volcano_clickData)
        return sd_table_content
    
    elif triggered_id == 'all-significant':        
        sd_table_content  = display_hist_n_table(False, True)
        return sd_table_content
    
    elif triggered_id == 'filter-button':        
        return display_options(None)
    
    elif triggered_id == 'pca-plot':
        sd_table_content  = display_hist_n_table(pca_clickData)
        return sd_table_content
    
    elif triggered_id == 'fg-table':
        fg_table_selected_row = str(fg_table_selected_row[0])
        sd_table_content  = display_hist_n_table(fg_table_selected_row)
        return sd_table_content
    
    elif triggered_id == 'variation-button':
        layout = display_variance_info()
        return layout
    
    elif triggered_id == 'export-button':
        global featurelistname, stand_ion_df
        error = ''
        try:
            export_rda(stand_ion_df)
            response = f'feature/{featurelistname}_response.csv'
            explanatory = f'feature/{featurelistname}_explanatory.csv'
        except Exception as e:
            response = 'Error'
            explanatory = 'Error'
            error = f'Error is: {e}'
        layout = html.Div([
                    dbc.Label('Data exported to:'),
                    html.Br(),
                    dbc.Label(response),
                    html.Br(),
                    dbc.Label(explanatory),
                    html.Br(),
                    dbc.Label(error)
                    ])
        return layout
    
    return dash.no_update  # Return no update if the callback was not triggered by the expected inputs

 
    
# Dynamic content module
###############################################################################
# Function to generate the standardized table. Each feature from a same ion group are standardized through samples
def create_sd_table(clickData):
    global G, pres_sample_threshold
    clicked_node_identifier = None
    if clickData:
        if isinstance(clickData, str): # if it is selected row from the feature group list
            clicked_node_identifier = int(fg_table_render.iloc[int(clickData)]['FG'])
        else:                    
            clicked_element = clickData['points'][0]
            if 'text' in clicked_element and 'FG/MFG:' in clicked_element['text']:
                clicked_node_identifier = int(clicked_element['text'].split('FG:')[1].split(' ')[0])
        if clicked_node_identifier:        
            for node in G.nodes():
                if G.nodes[node]['feature_group'] == clicked_node_identifier:
                    node_id = node # node_id is one of the feature in the node, used to find back all the other features of the node
            for component in nx.connected_components(G):
                if node_id in component:
                    connected_component = component
                    break
                
            global sd_table, all_final_mean, all_sample_mean, stat_result, sample_list
            sd_table, all_final_mean, all_sample_mean, stat_result = stat_n_assumptions(connected_component, sample_list, pres_sample_threshold)
            
            return True, clicked_node_identifier
        else:
            return False, clicked_node_identifier

# Only when clicking on Sig button over the volcano plot
def create_sd_table_significant():
    global volcano_feature_groups_significant, G, pres_sample_threshold
    if len (volcano_feature_groups_significant) > 0:
        FG_list = [fg.split('FG/MFG:')[1] for fg in volcano_feature_groups_significant] # take all feature groups identifiers which are significant
        node_id = []
        for fg in FG_list:
            for node in G.nodes(): # we need to know the node id, meaning the unique identifier in the network corresponding to a given feature present in a faature group
                if fg in str(G.nodes[node]['feature_group']):
                    node_id.append(node) # node_id is one of the feature in the node, used to find back all the other features of the node
                    break
        significant_component = []# a component of a network graph is what corresponds to feature group in MassLearn
        for n in node_id:
            for component in nx.connected_components(G):
                if n in component:
                    significant_component.extend(list(component))
                    break #the matching component is found
        global sd_table, all_final_mean, all_sample_mean, stat_result, sample_list
        sd_table, all_final_mean, all_sample_mean, stat_result = stat_n_assumptions(significant_component, sample_list, pres_sample_threshold)
        return True
    else:
        return False

# Function to create the specctra displayer
def spectra_plot():
    pass

# Function to resume main information about the feature group selected
def fg_informations(Sd):
    global project_loaded
    if not(project_loaded.meta):
        feature_dict = {str(row['feature']): (row['rt'], row['m/z']) for _, row in Sd.iterrows()}
        feature_df = pd.DataFrame([(feature, rt, mz) for feature, (rt, mz) in feature_dict.items()], columns=['Feature', 'rt', 'm/z'])
        esi_mode = set(list(project_loaded.template_esi_mode.values()))
        if len(esi_mode) == 2: # if there are neg and pos
            return pd.DataFrame({})
        if list(esi_mode)[0] == 'pos':
            mode = 'positive'
        else:
            mode = 'negative'
        adduct_df = pd.read_csv(f'./data/{mode}_mode_adducts.csv')
        feature_adduct_df = feature_df.copy()
        # Loop through each row in df2 and create a new column in df3 for each 'adduct'
        for index, row in adduct_df.iterrows():
            adduct_name = row['adduct']
            mass_change = row['mass_change']
            
            # Create new columns in df3 with adjusted 'm/z' values
            feature_adduct_df[adduct_name] = round(feature_df['m/z'] - mass_change, 4) # remove the adduct mass to find the original neutral mass
            feature_adduct_df['rt'] = feature_adduct_df['rt'].round(2)
        return feature_adduct_df
    else:
        return pd.DataFrame({})

# Function to display the feature table, with standardized values of height
def display_hist_n_table(clickData, Significant = False):
    global project_loaded
    # Logic to create Distance Matrix goes here
    feature_group = False
    sig = False
    hist_title = ''
    if Significant:
        sig = create_sd_table_significant()
        hist_title = 'All significants feature groups from volcano plot'
    elif clickData:
        feature_group, clicked_node_identifier = create_sd_table(clickData)
        hist_title = f"feature group {clicked_node_identifier} associated features"
    if feature_group or sig:
        global sd_table, all_final_mean, all_sample_mean, stat_result
        # Define the tabs
        stat_table = stat_result.copy(deep=True)
        sd = sd_table.copy(deep=True)
        if project_loaded.meta == False:
            sd = sd.drop(columns=['height'], axis=1)
        sd = sd.rename(columns={'std_v': 'values'})
        sd = sd.sort_values(by='m/z', ascending=False)
        fg_info = fg_informations(sd)
        
        stat_tab_content = html.Div([
                                dash_table.DataTable(
                                    data=stat_table.to_dict('records'),  # Convert DataFrame to dictionary
                                    columns=[{'name': i, 'id': i} for i in stat_table.columns],
                                    row_deletable=False,
                                    cell_selectable=False,
                                    style_table={'backgroundColor': '#f4f4f4'},  # Example: light gray table background
                                    style_header={
                                            'backgroundColor': 'rgb(0, 0, 0)',
                                            'fontWeight': 'bold'
                                        },
                                    style_cell={
                                        'backgroundColor': 'rgb(20, 20, 20)',  # Example: dark blue cell background
                                        'color': '#ffffff',            # Example: white text color
                                        'textAlign': 'left',
                                        'padding': '10px'},
                                    style_data={
                                        'borderColor': 'black',  # Example: black border for data cells
                                        'borderWidth': '1px',
                                        'borderStyle': 'solid'},
                                                    ),
                                    ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})
        table_tab_content = html.Div([
                                dbc.Table.from_dataframe(sd, striped=True, bordered=True, hover=True, size="sm"),
                                    ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})

        spectra_tab_content = html.Div([
                                    dcc.Graph(id='spectra-plot', style={"height": "400px"})
                                    ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})
        # TODO see https://dash-example-index.herokuapp.com/dash-download-component-app
        download_tab_content = html.Div([
                                    dbc.Table.from_dataframe(fg_info, striped=True, bordered=True, hover=True, size="sm", color="info"),
                                    ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})

        chromatogram_tab_content = html.Div([
                                    dcc.Loading(
                                        id="chromatogram-button-loading",
                                        type="default",
                                        children=html.Div(
                                            dbc.Button(
                                                "Display the features chromatogram",
                                                id="load-chromatograms",
                                                color="primary",
                                                outline=False,
                                            ),
                                            style={
                                                'display': 'flex',
                                                'justifyContent': 'center',
                                                'padding': '10px'
                                            }
                                        ),
                                    ),
                                    dcc.Loading(
                                        id="chromatogram-loading-wrapper",
                                        type="default",
                                        children=html.Div(id="chromatogram-container"),
                                    ),
                                ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})

        layout = html.Div([
                    html.Div([
                        html.H4(hist_title, style={'textAlign': 'left', 'paddingBottom': '10px'}),
                    
                        html.Div([
                            # Histogram
                            dcc.Graph(id='abundance-histogram', style={'width': '100%', 'display': 'inline-block', 'height': '370px', 'margin-top': '10px'}),
                            dbc.Tooltip(
                                "Statistical test are made based on the mean of the Standardized values for each sample across all features in the feature group. A Benjamini-Holchberg correction is made taking all feature groups tested.",
                                target="abundance-histogram",  # ID of the component to which the tooltip is attached
                                placement="right"),
                
                            # Level Selection
                            html.Div([
                                dbc.Label("Labelling option", style={'display': 'inline-block'}),
                                dbc.RadioItems(
                                    options=dropdown_items,
                                    value=dropdown_items[0]['value'],
                                    id="hist-level",
                                    inline=True,
                                    style={'display': 'inline-block', 'verticalAlign': 'top', 'margin-left': '10px'})
                                    ], style={'width': '100%'})  
                                ]),
                                ], style={'display': 'inline-block', 'width': '35%', 'margin-left': '10px',  'margin-right': '10px'}),
                    html.Div([
                                dbc.Tabs(
                                    [   dbc.Tab(stat_tab_content, label="Statistical analysis", tab_id="tab-1"),
                                        dbc.Tab(table_tab_content, label="Datatable", tab_id="tab-2"),
                                        dbc.Tab(download_tab_content, label="Feature group informations", tab_id="tab-4", id= 'fg-info-tab'),
                                        dbc.Tab(chromatogram_tab_content, label="Chromatograms", tab_id="tab-5"),
                                    ], active_tab="tab-1"
                                        ),
                                dbc.Tooltip("Find all features from the feature group their potential neutral mass in Da, depending on the most abundant adduct from the ESI mode.",
                                target="fg-info-tab",  # ID of the component to which the tooltip is attached
                                placement="right"),
                                    ], style={'display': 'inline-block', 'width': '60%', 'vertical-align': 'top', 'margin-left': '10px', 'margin-top': '10px'}),
                            ])
        return layout
    else:
        return html.Div([])

# Function to display the feature table, with standardized values of height
def display_variance_info():
    global stand_ion_df, ion_df, project_loaded
    variance_df, variance_df_str, levels_coef_determination_dict = make_variance_df(stand_ion_df)
       
    var = html.Div([dbc.Table.from_dataframe(variance_df_str, striped=True, bordered=True, hover=True, size="sm"),
                                ], style={ 'margin-right': '10px', 'maxHeight': '400px', 'overflow': 'auto', 'margin-top': '10px'})

    var = dash_table.DataTable(
        data=variance_df_str.to_dict('records'),  # Convert DataFrame to dictionary
        columns=[{'name': i, 'id': i} for i in variance_df_str.columns],
        sort_action="native",
        page_size=10,
        row_selectable="single",
        cell_selectable=False,
        sort_by=[{'column_id': 'Feature', 'direction': 'desc'}],
        style_table={
            'backgroundColor': 'rgb(250, 250, 250)',
            'overflowX': 'auto'},  # Example: light gray table background
        style_header={
            'fontFamily': '"Helvetica Neue", Arial, sans-serif',
            'backgroundColor': 'rgb(235, 235, 250)',
            'color': 'rgb(0, 0, 0)',
            'fontWeight': 'bold'},
        style_cell={
            'fontFamily': '"Helvetica Neue", Arial, sans-serif',
            'backgroundColor':'rgb(250, 250, 250)',  # Example: dark blue cell background
            'color': 'rgb(0, 0, 0)',            # Example: white text color
            'textAlign': 'left',
            'overflow': 'hidden',
            'fontSize': '12px',
            'textOverflow': 'ellipsis',
            'padding': '5px'},
        style_data={
            'borderColor': 'rgb(102, 102, 102)',  # Example: black border for data cells
            'borderWidth': '1px',
            'borderStyle': 'solid'},
        style_data_conditional=[{
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(245, 245, 245)' }]
                        ),
    description = html.Div([
        html.P(
            "The variation table summarizes how much each treatment level explains the standardized intensity of every feature (R² values).",
            style={'marginBottom': '4px'}),
        html.P(
            "Higher percentages highlight feature groups that discriminate best between conditions; use the row selector to inspect specific features.",
            style={'marginBottom': '8px'}),
    ])

    return html.Div(
        [description, var],
        style={
            'margin-right': '10px',
            'maxHeight': '420px',
            'overflow': 'auto',
            'margin-top': '10px',
        },
    )

    
# Callback to update the hist once the dynamic content is clicked
@callback(
    Output('abundance-histogram', 'figure'),
    Input('hist-level', 'value')
  # Assuming the dynamic content is stored in a component with ID 'dynamic-content'
)
def update_histogram(Selected_level):
    hist_figure = create_histogram_figure(Selected_level)
    return hist_figure


# Function to create histogram. Histogram update when user click on possible levels. Each dot is a unique feature, samples share same dot color because they might have all the same standardized value if the features from the ion correlate
def create_histogram_figure(Level):
    Nature = 'Area'
    global sd_table, all_final_mean, all_sample_mean
    final_mean = all_final_mean[Level]
    sample_mean = all_sample_mean[Level]
    if stat_result.loc[stat_result['Level'] == Level,].any().any(): # if there are stat test for such level
        pvalue = stat_result.loc[stat_result['Level'] == Level, 'Test_p-value'].values[0]
        assumption_validity = stat_result.loc[stat_result['Level'] == Level, 'Assumption_validity'].values[0]
        test_type = stat_result.loc[stat_result['Level'] == Level, 'Test_type'].values[0]
        non_param = stat_result.loc[stat_result['Level'] == Level, 'Non_parametric_p-value'].values[0]
        non_param = f' --  Non-parametric p-value: {non_param}'
    else:
        pvalue = 'Not enough data'
        assumption_validity = 'Not enough data'
        test_type = 'Not enough data'
    
    final_mean['combined'] = final_mean.apply(lambda row: f"{row['std_v']}<br>{row[f'{Level}_level_category']}", axis=1)

    fig = go.Figure()
    for i, row in final_mean.iterrows():
        fig.add_trace(go.Bar(
            x=[row[f'{Level}_level_category']],
            y=[row['std_v']],
            textposition='none',
            marker=dict(color='darkgrey'),
            name=row['combined'],  # Use the combined column for the legend text
            showlegend=False  # TO REMOVE
        ))

    # Create a color map for unique sample names
    unique_samples = sd_table['sample'].unique()
    colors = px.colors.qualitative.Plotly
    color_map = {sample: colors[i % len(colors)] for i, sample in enumerate(unique_samples)}

    # Assuming you have a way to assign each sample a unique identifier
    sample_ids = {sample: i+1 for i, sample in enumerate(unique_samples)}
    
    
    # Modify the plotting part to use sample_ids for marker names and detailed info in hover text
    for level_cat in final_mean[f'{Level}_level_category'].unique():
        subset = sd_table[sd_table[f'{Level}_level_category'] == level_cat]
        for sample in subset['sample'].unique():
            sample_subset = subset[subset['sample'] == sample]
            if len(sample) > 12: # if too many characters
                samplename = sample_ids[sample]
            else:
                samplename = sample

            
            # Add data points for mean valeus of all feature list for each sample
            fig.add_trace(go.Scatter(
                x=[level_cat],
                y=sample_subset['std_v'],
                mode='markers',
                marker=dict(color=color_map[sample], symbol='diamond-wide-open', size=4),
                name=samplename,  # Use the unique identifier for display
                hoverinfo='text',
                text=[f'{sample} - feature {fg} - {val}' for val, fg in zip(sample_subset['std_v'], sample_subset['feature'])],
                showlegend=False  # TO REMOVE# The detailed name and info will show on hover
            ))
            
            # Adding mean value as special dot, with hover text for detailed info
            mean_value = round(sample_mean[sample_mean['sample'] == sample]['std_v'].values[0], 2)
            fig.add_trace(go.Scatter(
                x=[level_cat],
                y=[mean_value],
                mode='markers',
                marker=dict(color=color_map[sample], symbol='diamond-wide-dot', size=10),
                name=f'Mean {samplename}',  # Display 'Mean' + unique identifier
                hoverinfo='text',
                text=[f'{sample} - mean: {mean_value}'],
                showlegend=False  # TO REMOVE
                # Detailed sample name and mean value on hover
            ))
            
    
    fig.update_layout(title=f'Abundance histogram - Level: {Level}',
                      xaxis_title='Level Category',
                      yaxis_title=f'Average Standardized {Nature}',
                      margin=dict(l=20, r=0, b=100, t=40),)
    
    fig.add_annotation(
    text=f'Type: {test_type}<br>P value: {pvalue}<br>Assumptions validity: {assumption_validity} {non_param if assumption_validity == "no" else ""}', # Your custom text
    xref="paper", yref="paper",  # These reference settings ensure positioning relative to the figure
    x=0.1, y=-0.4,  # Adjust these values to position your text
    showarrow=False,  # Set to False so that no arrow is shown
    font=dict(
    size=11,  # Font size of the text
    color="grey"  # Font color
    ),
    align="left"
    )

    return fig


def _build_chromatogram_layout():
    global sd_table, shape_record_store
    if sd_table is None or len(sd_table) == 0:
        return html.Div("No feature group is currently selected.")

    if not shape_record_store:
        return html.Div("Chromatogram files are unavailable for this project.")

    if 'shape_id' not in sd_table.columns:
        return html.Div("No chromatogram identifiers were found for the current selection.")

    feature_shapes: Dict[str, List[dict]] = {}
    feature_meta: Dict[str, dict] = {}

    for _, row in sd_table.iterrows():
        feature_id = str(row.get('feature'))
        feature_meta.setdefault(
            feature_id,
            {
                'rt': row.get('rt'),
                'm/z': row.get('m/z'),
            },
        )
        shape_id = row.get('shape_id')
        if pd.isna(shape_id):
            continue
        record = shape_record_store.get(str(shape_id))
        if not record:
            continue
        shape_payload = record.get('shape')
        if (
            not isinstance(shape_payload, (list, tuple))
            or len(shape_payload) != 2
            or len(shape_payload[0]) < 2
        ):
            continue
        feature_shapes.setdefault(feature_id, []).append(
            {
                'sample': record.get('sample') or 'Unknown sample',
                'ms_level': record.get('ms_level'),
                'shape': shape_payload,
            }
        )

    if not feature_shapes:
        return html.Div("No chromatogram traces are available for the selected features.")

    def _feature_title(fid: str) -> str:
        meta = feature_meta.get(fid, {})
        mz_val = meta.get('m/z')
        rt_val = meta.get('rt')
        mz_str = f"m/z {mz_val:.4f}" if pd.notna(mz_val) else "m/z N/A"
        rt_str = f"rt {rt_val:.2f}s" if pd.notna(rt_val) else "rt N/A"
        return f"Feature {fid} ({mz_str}, {rt_str})"

    feature_order = sorted(
        feature_shapes.keys(),
        key=lambda fid: feature_meta.get(fid, {}).get('m/z') or 0,
        reverse=True,
    )

    chart_cards = []
    for fid in feature_order:
        traces = feature_shapes[fid]
        fig = go.Figure()
        for trace in sorted(traces, key=lambda t: t['sample']):
            rts, intensities = trace['shape']
            fig.add_trace(
                go.Scatter(
                    x=rts,
                    y=intensities,
                    mode='lines',
                    name=str(trace['sample']),
                    hovertemplate='Sample: %{text}<br>RT: %{x}<br>Intensity: %{y}<extra></extra>',
                    text=[trace['sample']] * len(rts),
                )
            )
        fig.update_layout(
            title=_feature_title(fid),
            margin=dict(l=10, r=10, t=40, b=30),
            height=280,
            legend=dict(orientation='h', y=-0.25),
            xaxis_title='Retention time',
            yaxis_title='Intensity',
        )
        chart_cards.append(
            html.Div(
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
                style={'marginBottom': '12px'},
            )
        )

    left_column = html.Div(
        chart_cards[0::2],
        style={
            'width': '50%',
            'maxHeight': '380px',
            'overflowY': 'auto',
            'paddingRight': '8px',
        },
    )
    right_column = html.Div(
        chart_cards[1::2],
        style={
            'width': '50%',
            'maxHeight': '380px',
            'overflowY': 'auto',
            'paddingLeft': '8px',
        },
    )

    return html.Div(
        [left_column, right_column],
        style={'display': 'flex', 'width': '100%'},
    )

@callback(
    Output('chromatogram-container', 'children'),
    Output('load-chromatograms', 'children'),
    Output('load-chromatograms', 'disabled'),
    Input('load-chromatograms', 'n_clicks'),
    prevent_initial_call=True,
)
def _render_feature_chromatograms(n_clicks):
    if not n_clicks:
        raise PreventUpdate()

    layout = _build_chromatogram_layout()

    button_child = dbc.Spinner(size="sm", color="light")
    button_disabled = True
    if layout:
        button_child = "Display the features chromatogram"
        button_disabled = False

    return layout, button_child, button_disabled


# Filtering options of the Network graph
###########################################################################
def display_options(Click = None): # Click is a None value when the filtering option panel is opened, to prevent the Message canva to open first
    global all_treatments, treatment_selection
    checklist = dbc.Col(                            
                    html.Div([      
                        dbc.Offcanvas(id='update-message-offcanvas', title="Update Message", is_open=False),
                        dbc.Label("Treatment and other"),
                        dbc.Checklist(
                            options=all_treatments,
                            value=treatment_selection,
                            id="filter-treatment"),
                        dbc.Tooltip(
                                "If unchecked, all samples related to this treatment are deleted to make the Network of LCMS features",
                                target="filter-treatment",  # ID of the component to which the tooltip is attached
                                placement="right"),
                            ]),
                        width = 2)
    
    layout = html.Div([
                html.H3(["Feature network filtering options"]),
                checklist
                      ])
    return layout   

# Function to remove or add (filter) treatments from the filtering options. Treatment variable are all treatment which are selected
def filter_sample(Treatments):
    global featurelist, all_treatments, sample_list, preprocessed_df, featurelist
    global dropdown_items, dropdown_items_binary
    raw_values = [t['value'] for t in all_treatments] # here are all the treatments no matter the levels, in a list
    to_filter = []
    for t in raw_values:
        if t not in Treatments:
            to_filter.append(t)

    # Iterate over each sample in sample_list and check if a treatment is present
    sample_removed = [] 
    for column in sample_list.columns:
        if sample_list[column].isin(to_filter).any():
            sample_removed.append(column)
    
    sample_list.drop(sample_removed, axis=1, inplace=True)            
    preprocessed_df.drop(sample_removed, axis=1, inplace=True)
    stand_preprocessed_df.drop(sample_removed, axis=1, inplace=True)
    featurelist = featurelist[~featurelist['sample'].isin(sample_removed)]
    _refresh_feature_shape_vectors(featurelist)
    dropdown_items, dropdown_items_binary  = dropdowns(sample_list)
    

# Callback to update the hist once the dynamic content is clicked
@callback(
    Output('update-message-offcanvas', 'children'),
    Output('update-message-offcanvas', 'is_open'),
    Input('filter-treatment', 'value')
  # Assuming the dynamic content is stored in a component with ID 'dynamic-content'
)
def update_samples(Selected_treatments):
    trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    global sample_list, dropdown_items, dropdown_items_binary
    global preprocessed_df, stand_preprocessed_df
    global featurelist
    global treatment_selection
    
    if trigger == 'filter-treatment':
        # Open the offcanvas with the message
        treatment_selection = Selected_treatments          
        
        # Step 1: Reset the variables
        sample_list = sample_list_raw.copy(deep=True)
        preprocessed_df = preprocessed_df_raw.copy(deep=True)
        featurelist = featurelist_raw.copy(deep=True)
        stand_preprocessed_df = stand_preprocessed_df_raw.copy(deep=True)
        dropdown_items, dropdown_items_binary = dropdowns(sample_list)
        _refresh_feature_shape_vectors(featurelist)
        # Step 2: Adapt sample list
        filter_sample(Selected_treatments)
        
        message = "Click on the red Update Graph button to apply changes."
    
        return message, True  
    else:
        # If no selection, do not open the offcanvas
        return dash.no_update, False          


# PCA module
###############################################################################
def tuple_to_hex(Color_tuple):
    return "#{:02x}{:02x}{:02x}".format(int(Color_tuple[0] * 255), 
                                        int(Color_tuple[1] * 255), 
                                        int(Color_tuple[2] * 255))

# Function to manage pca color
def generate_colors(Num_colors):
    colormap = plt.cm.tab20 # or any other colormap like "viridis", "plasma", "inferno", "magma", "cividis", etc.
    colors = [tuple_to_hex(colormap(i)) for i in range(colormap.N)]
    if Num_colors <= len(colors):
        return colors[:Num_colors]
    else:
        return colors * (Num_colors // len(colors)) + colors[:Num_colors % len(colors)]

# Callback to generate 'pca-plot' after 'network-graph' is rendered
@callback(
    [Output('pca-plot', 'figure'),
     Output('dimension-button', 'children')],
    [Input('intermediate-signal-pca', 'children'),  # Signal from network-graph rendering
     Input('pca-type', 'value'),       # Various states
     Input('pca-level', 'value'),
     Input('pca-title', 'n_submit'),
     Input('pca-nature', 'value'),
     Input('dimension-button', 'n_clicks'),],
    [State('pca-title', 'value'),
     State('dimension-button', 'children')],
    prevent_initial_call = True
)
def show_pca(intermediate_signal, pca_type, pca_level, title_n_submit, pca_nature, dimension_n_clicks, pca_title, button_dimension):
    global validity, network_updated, current_dimension
    trigger = callback_context.triggered[0]['prop_id'].split('.')[0]
   
    if not validity:
        raise PreventUpdate()

    # Logic to create PCA plot goes here
    # Use callback_context to determine which input was triggered
    if trigger == 'dimension-button':
        if current_dimension == '2D':
            current_dimension = '3D'
            button_dimension = '2D'
        else:
            current_dimension = '2D'
            button_dimension = '3D'
    pca_figure = create_pca_figure(pca_type, pca_level, pca_title, pca_nature, current_dimension)
    return pca_figure, button_dimension

    

def create_pca_figure(Type=1, Level='1', Title='Title', Nature='f', Dimension='3D'):
    global ion_df, stand_ion_df, sample_list
    
    if Dimension == '2D':
        Dimension = 2
    else:
        Dimension = 3
        
    if Title is None:
        Title = 'No title'

    if Type == 1:  # i.e., sample
        if Nature == 'f':
            table = ion_df.drop(columns=['m/z', 'rt', 'FG', 'Size'])
            #table = (table - table.mean()) / table.std() # standardized table
        else:
            table = stand_ion_df.drop(columns=['m/z', 'rt', 'FG', 'Size'])
        
        table = table.transpose()
        labels = []
        for s in table.index:
            lab = sample_list.loc[Level, s]
            labels.append(lab)
    else:
        labels = ['FG or MFG' for _ in range(len(ion_df))]
        if Nature == 'f':
            table = ion_df
        else:
            table = stand_ion_df
        feature_name = [f'mz:{mz} RT:{round(rt, 2)} FG/MFG:{int(fg)}' for mz, rt, fg in zip(ion_df['m/z'], ion_df['rt'], ion_df['FG'])]
        table.index = feature_name
        table = table.drop(columns=['m/z', 'rt', 'FG', 'Size'])

    pca = PCA(n_components=Dimension)
    pca_result = pca.fit_transform(table)

    # Create a dataframe from PCA result
    columns = [f'pc{i+1}' for i in range(Dimension)]
    pca_df = pd.DataFrame(data=pca_result, columns=columns)
    pca_df['y'] = pd.Series(labels)

    # Add a new column with the index names from the original dataframe
    indexlist = [[str(x) for x in j] for j in table.index.to_list()]
    if Type == 1:
        pca_df['index_name'] = [' '.join(i) for i in indexlist]
    else:
        pca_df['index_name'] = [''.join(i) for i in indexlist]

    # Create color scale based on unique labels
    num_labels = len(np.unique(labels))
    colorscale = px.colors.qualitative.Plotly if num_labels <= 10 else generate_colors(num_labels)

    # Assign color for each unique label
    colors_dict = {str(label): colorscale[i % len(colorscale)] for i, label in enumerate(np.unique(labels))}
    pca_df['color'] = pca_df['y'].map(colors_dict)

    # Define marker symbols
    symbols = [
    # Basic Symbols
    'circle', 'square', 'diamond', 'cross', 'x', 'pentagon', 'hexagon', 'hexagon2', 'octagon', 'star',
    'hexagram', 'star-triangle-up', 'star-triangle-down', 'star-square', 
    'star-diamond', 'diamond-tall', 'diamond-wide', 'hourglass', 'bowtie',

    # Open Symbols
    'circle-open', 'square-open', 'diamond-open', 'cross-open', 'x-open', 
    'pentagon-open', 'hexagon-open', 'hexagon2-open', 'octagon-open', 'star-open', 
    'hexagram-open', 'star-triangle-up-open', 'star-triangle-down-open', 
    'star-square-open', 'star-diamond-open', 'diamond-tall-open', 'diamond-wide-open', 
    'hourglass-open', 'bowtie-open']

    # Assign symbols to each label
    symbols_dict = {str(label): symbols[i % len(symbols)] for i, label in enumerate(np.unique(labels))}
    pca_df['symbol'] = pca_df['y'].map(symbols_dict)

    if Type == 1:
        legend_t = f'Labels for {Level}'
        label_indication = "Dots label: your sample names"
    else:
        legend_t = ""
        label_indication = "Each dot is a FG/MFG of features.<br>mz for m/z (Da), RT for retention time (min), FG/MFG for (meta) feature group (index number)"

    if Dimension == 3:
        # Create 3D scatter plot
        scatter = go.Scatter3d(
            x=pca_df['pc1'],
            y=pca_df['pc2'],
            z=pca_df['pc3'],
            mode='markers',
            marker=dict(
                size=12,
                color=pca_df['color'],  # Use color column
                symbol=pca_df['symbol'],  # Use symbol column
                opacity=0.8
            ),
            text=pca_df['index_name'],  # Use the index_name column for the hover text
            hoverinfo='text',
            showlegend=False)

        # Create legend items
        legend_items = [
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol=symbols_dict[str(label)]),
                name=label,
            )
            for label, color in colors_dict.items()]

        fig = go.Figure(data=[scatter] + legend_items)

        # Set layout with explained variance ratio and axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)",
            ),
            margin=dict(l=0, r=0, b=40, t=0),
            legend_title=legend_t,
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        fig.add_annotation(
            text=label_indication,  # Your custom text
            xref="paper", yref="paper",  # These reference settings ensure positioning relative to the figure
            x=0.05, y=-0.1,  # Adjust these values to position your text
            showarrow=False,  # Set to False so that no arrow is shown
            font=dict(
                size=11,  # Font size of the text
                color="grey"  # Font color
            ),
            align="left"
        )
    else:
        # Create 2D scatter plot
        scatter = go.Scatter(
            x=pca_df['pc1'],
            y=pca_df['pc2'],
            mode='markers',
            marker=dict(
                size=10,
                color=pca_df['color'],  # Use color column
                symbol=pca_df['symbol'],  # Use symbol column
                opacity=0.8
            ),
            text=pca_df['index_name'],  # Use the index_name column for the hover text
            hoverinfo='text',
            showlegend=False
        )

        # Create legend items
        legend_items = [
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol=symbols_dict[str(label)]),
                name=label,
            )
            for label, color in colors_dict.items()
        ]

        fig = go.Figure(data=[scatter] + legend_items)

        # Set layout with explained variance ratio and axis labels
        fig.update_layout(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            margin=dict(l=10, r=10, b=80, t=60),
            legend_title=legend_t,
            legend=dict(
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            )
        )
        fig.add_annotation(
            text=label_indication,  # Your custom text
            xref="paper", yref="paper",  # These reference settings ensure positioning relative to the figure
            x=0.05, y=-0.25,  # Adjust these values to position your text
            showarrow=False,  # Set to False so that no arrow is shown
            font=dict(
                size=11,  # Font size of the text
                color="grey"  # Font color
            ),
            align="left"
        )

    fig.update_layout(
        title={
            'text': f'PCA Analysis - {Title}',
            'y': 0.95,  # Adjust this value as needed
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig


# Volcano plot module
###############################################################################
# Callback to generate 'pca-plot' after 'network-graph' is rendered
@callback(
    [Output('volcano-plot', 'figure'),
     Output('chi-square-res', 'children')],
    [Input('intermediate-signal-volcano', 'children'),  # Signal from network-graph rendering
     Input('volcano-pvalue', 'value'),       # Various states
     Input('volcano-level', 'value'),
     Input('volcano-title', 'n_submit')],
    State('volcano-title', 'value'),
    State('volcano-table-progress', 'value'),
    prevent_initial_call = True
)
def show_volcano(intermediate_signal, volc_pvalue, volc_level, title_n_submit, volc_title, volc_progress):
    if volc_progress < 100:
        raise PreventUpdate()        
    else:
        volcano_figure = create_volcano_plot(volc_pvalue, volc_level, volc_title)
        return volcano_figure


def create_volcano_plot(P_value = 0.05, Level = '', Title = ''):
    global volcano_feature_groups_significant
    # Extract samples for the specified treatment condition
    condition_samples =[]
    if ' vs ' in Level:
        level_name = Level.split(' - ')[0] # E.g Line
        treatments = Level.split(' - ')[1].split(' vs ') # e.g wt and mutant
        for t in treatments:
            condition_samples.append(sample_list.transpose().loc[sample_list.transpose()[level_name] == t,].index.to_list())
        log_fold_changes, p_values, volcano_feature_groups, volcano_feature_groups_significant = calculate_log_fold_change_and_p_values(condition_samples[0], condition_samples[1])
        
    elif len(sample_list.loc[Level].unique()) == 2:
        treatments = []
        for t in sample_list.loc[Level].unique():
            treatments.append(t)
            condition_samples.append(sample_list.transpose().loc[sample_list.transpose()[Level] == t,].index.to_list())
        log_fold_changes, p_values, volcano_feature_groups, volcano_feature_groups_significant = take_pvalue_and_calculate_lfc(Level, condition_samples[0], condition_samples[1])
    
    if condition_samples != []:
        # Chi square part        
        positive_fold_changes = np.sum(log_fold_changes > 0) # Classify features based on log fold changes
        negative_fold_changes = np.sum(log_fold_changes < 0)
        observed = np.array([positive_fold_changes, negative_fold_changes]) # Create observed and expected counts
        expected = np.array([len(log_fold_changes) / 2, len(log_fold_changes) / 2])
        try:
            chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
            chi_square_success = True
        except ValueError as e:
            chi2, p_value = 0, 1
            chi_square_success = False
            error_message = str(e)
        
        # Display the results
        total_features = len(log_fold_changes)
        positive_percentage = (positive_fold_changes / total_features) * 100
        negative_percentage = (negative_fold_changes / total_features) * 100

        result_sentence = (
            f"{total_features} FG, {positive_percentage:.2f}% pos, {negative_percentage:.2f}% neg. Chi-square test of {chi2:.2f}, p = {p_value:.3f}."
        )
        if chi_square_success:
            pass
        else:
            result_sentence += (
                f"Value error: {error_message}."
        )

        fig = go.Figure()    
        # Plot each point
        fig.add_trace(go.Scatter(
            x=log_fold_changes,
            y=-np.log10(p_values),
            mode='markers',
            marker=dict(
                color=np.where(p_values < P_value, 'red', 'blue'),  # Highlight significant points
                size=7
            ),
            text=[f'LFC: {format_value(lfc)}, P-value: {format_value(p)}<br>{ion}' for lfc, p, ion in zip(log_fold_changes, p_values, volcano_feature_groups)],  # Hover text
            hoverinfo='text'
        ))
        
        # Enhance the layout
        fig.update_layout(
            xaxis_title=f' {treatments[0]} <-- LFC --> {treatments[1]}',
            yaxis_title='-Log10(P-value)',
            template='plotly_white'
        )    
        fig.update_layout(
        margin=dict(l=40, r=10, b=100, t=50),
        title={
            'text': f'Volcano plot - {Title}',
            'y': 0.95,  # Adjust this value as needed
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
        fig.add_annotation(
        text='Each dot is a FG/MFG of features.<br>LFC for Log Fold Change, P-Value is the significance level<br>mz for m/z (Da), RT for retention time (min), FG for feature group',  # Your custom text
        xref="paper", yref="paper",  # These reference settings ensure positioning relative to the figure
        x=0, y=-0.35,  # Adjust these values to position your text
        showarrow=False,  # Set to False so that no arrow is shown
        font=dict(
        size=11,  # Font size of the text
        color="grey"  # Font color
        ),
        align="left"
        )
        return fig, result_sentence
    else:
        result_sentence = ''
        fig = go.Figure()
        fig.add_annotation(
            text='Need two paramters to compare.<br>Check "More filtering"',
            x=2,  # Center horizontally
            y=1.5,  # Center vertically
            showarrow=False,
            font=dict(
                size=14,  # Adjust the font size as needed
                color="red",  # Adjust the text color as needed
                family="Arial"  # Adjust the font family as needed
            ),
            align="center"  # Center-align the text
        )
        return fig, result_sentence
        

# Function to have lfc at an existing  binay level from all the features present in fg table
def take_pvalue_and_calculate_lfc(Level, Group1_samples, Group2_samples, Small_constant=1e-4):
    global ion_df, fg_table
    
    feature_groups = fg_table.loc[fg_table[f'{Level} t-test p'] != '-', 'FG']
    feature_groups.sort_values(inplace=True)
    
    reduced_fg_table = fg_table.loc[fg_table[f'{Level} t-test p'] != '-', [f'{Level} t-test p', 'FG']] # take p values from fg_table
    reduced_fg_table.sort_values(by='FG', inplace=True) # sort by FG index number
    p_values = reduced_fg_table[f'{Level} t-test p']
    feature_groups = reduced_fg_table['FG']
    
    Ion_df = ion_df.copy(deep=True)
    Ion_df = Ion_df.loc[Ion_df['FG'].isin(feature_groups),]
    Ion_df.sort_values(by='FG', inplace=True) # sort by FG index number
    Ion_df_reduced = ion_df.drop(columns = ['m/z', 'rt', 'Size', 'FG'])
    
    # Subset the DataFrame for each group
    group1 = Ion_df_reduced[Group1_samples]
    group2 = Ion_df_reduced[Group2_samples]
    
    # Calculate mean for each feature in each group
    mean_group1 = group1.mean(axis=1) + Small_constant
    mean_group2 = group2.mean(axis=1) + Small_constant

    # Calculate log fold changes and p-values
    log_fold_changes = np.log2(mean_group2 / mean_group1)    
    
    volcano_feature_groups = [f'mz:{mz} RT:{round(rt, 2)} FG/MFG:{int(fg)}' for mz, rt, fg, lfc in zip(Ion_df['m/z'], Ion_df['rt'], Ion_df['FG'], log_fold_changes) ]
    # volcano group significant is for the histogram grouping all significant features
    volcano_feature_groups_significant = [f'mz:{mz} RT:{round(rt, 2)} FG/MFG:{int(fg)}' for mz, rt, fg, lfc, pval in zip(Ion_df['m/z'], Ion_df['rt'], Ion_df['FG'], log_fold_changes, p_values) if abs(lfc) >= 1 and pval <= 0.05 ]
    
    return log_fold_changes, p_values, volcano_feature_groups, volcano_feature_groups_significant


# Function obvious. Can only compare two groups. The levels with more than 2 groups are not displayed in the volcano plot tool option
def calculate_log_fold_change_and_p_values(Group1_samples, Group2_samples, Small_constant=1e-4):
    global ion_df
    Ion_df = ion_df.copy(deep=True)
    Ion_df_reduced = ion_df.drop(columns = ['m/z', 'rt', 'Size', 'FG'])
    # Subset the DataFrame for each group
    group1 = Ion_df_reduced[Group1_samples]
    group2 = Ion_df_reduced[Group2_samples]
    
    # Calculate mean for each feature in each group
    mean_group1 = group1.mean(axis=1) + Small_constant
    mean_group2 = group2.mean(axis=1) + Small_constant

    # Calculate log fold changes and p-values
    log_fold_changes = np.log2(mean_group2 / mean_group1)
    
    # old code:
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     # Calculate log_fold_changes with condition
    #     log_fold_changes = np.where((mean_group1 == 0) | (mean_group2 == 0), 0, np.log2(mean_group2 / mean_group1))
    
    p_values = []
    exclusion_list = []
    for ix, lfc in enumerate(log_fold_changes):
        group1_values = group1.loc[ix]
        group2_values = group2.loc[ix]
        if len(group1_values) >= 3 and len(group2_values) >= 3:
            p_value = scipy.stats.ttest_ind(group1_values, group2_values, equal_var=False).pvalue
            if np.isnan(p_value):
                exclusion_list.append(ix)
            else:
                p_values.append(p_value)
                exclusion_list.append(0)            
        else:
            exclusion_list.append(ix)

    Ion_df = Ion_df[~Ion_df.index.isin(exclusion_list)]
    log_fold_changes_corrected = log_fold_changes.drop(exclusion_list)
    p_values = np.array(p_values)
    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]
    
    volcano_feature_groups = [f'mz:{mz} RT:{round(rt, 2)} FG:{int(fg)}' for mz, rt, fg, lfc in zip(Ion_df['m/z'], Ion_df['rt'], Ion_df['FG'], log_fold_changes_corrected) ]
    # volcano group significant is for the histogram grouping all significant features
    volcano_feature_groups_significant = [f'mz:{mz} RT:{round(rt, 2)} FG:{int(fg)}' for mz, rt, fg, lfc, pval in zip(Ion_df['m/z'], Ion_df['rt'], Ion_df['FG'], log_fold_changes_corrected, corrected_p_values) if lfc >= 1 and pval <= 0.05 ]
    
    return log_fold_changes_corrected, corrected_p_values, volcano_feature_groups, volcano_feature_groups_significant

# Function to adjust the volcano plot values format
def format_value(Val):
    Val = float(Val)
    if Val < 1:
        # Find the position of the first non-zero digit after the decimal
        if Val != 0:
            decimal_part = str(Val).split('.')[1]  # Get decimal part as string
            first_non_zero = next((i for i, d in enumerate(decimal_part) if d != '0'), len(decimal_part))
            return f"{Val:.{first_non_zero + 2}f}"
        else:
            return f"{Val:.1f}"
    else:
        return f"{Val:.1f}"
    
# pls-da module 
###############################################################################        
# Callback to generate 'plsda-plot' after 'network-graph' is rendered
@callback(
    Output('plsda-plot', 'figure'),
    [Input('intermediate-signal-plsda', 'children'),  # Signal from network-graph rendering
     Input('plsda-level', 'value'),
     Input('plsda-title', 'n_submit'),
     Input('plsda-nature', 'value')],
    State('plsda-title', 'value'),
    prevent_initial_call = True
)
def show_plsda(intermediate_signal, plsda_level, titble_submit, plsda_nature, plsda_title):
    global network_updated

    plsda_figure = create_pls_da_plot(plsda_level, plsda_title, plsda_nature)
    return plsda_figure

def calculate_classification_accuracy(pls, X, y):
    # Predict the group labels
    y_pred = pls.predict(X)
    # Convert predictions to binary labels based on a threshold (e.g., 0.5 for binary)
    y_pred_labels = (y_pred > 0.5).astype(int)
    # Calculate and return accuracy
    accuracy = np.mean(y_pred_labels == y)
    return accuracy

def your_p_value_calculation_function(X, y, n_components=2, n_permutations=100):
    # Original model fit
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    original_metric = calculate_classification_accuracy(pls, X, y)
    
    # Permutation test
    permuted_metrics = []
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        pls.fit(X, y_permuted)
        permuted_metric = calculate_classification_accuracy(pls, X, y_permuted)
        permuted_metrics.append(permuted_metric)
    
    # Calculate p-value
    p_value = np.mean(np.array(permuted_metrics) >= original_metric)
    return p_value

def perform_pls_da(X, y, n_components=2):
    # Initialize PLS model with specified number of components
    pls = PLSRegression(n_components=n_components)

    # Fit the model
    pls.fit(X, y)
    
    # # Compute total variance in X
    # total_variance = np.var(X, axis=0).sum()
    
    # # Compute variance explained by each component
    # variance_explained = []
    # for i in range(n_components):
    #     # Extract scores for component i
    #     scores = pls.x_scores_[:, i]
        
    #     # Compute variance for component i
    #     component_variance = np.var(scores, axis=0)
        
    #     # Compute percentage of total variance explained by component i
    #     variance_explained.append((component_variance / total_variance) * 100)

    # Get the scores
    scores = pls.transform(X)

    # Get the explained variance
    explained_variance = np.var(scores, axis=0) / np.var(X, axis=0).sum()

    # Return the scores and the variance explained
    return scores, explained_variance

def create_pls_da_plot(Level, Title, Nature, p_value=None):    
    global stand_ion_df, ion_df
    if len(sample_list.loc[Level].unique()) > 1: # is there are 2 treatments/conditions
        if Nature == 'f':
            temp_ion_df = ion_df.copy(deep=True)
        else:
            temp_ion_df = stand_ion_df.copy(deep=True)
        
        treatment_samples =[]
        for t in sample_list.loc[Level].unique():
            treatment_samples.append(sample_list.transpose().loc[sample_list.transpose()[Level] == t,].index.to_list())
        samples = {}
        group_labels = []
        for s, group in zip(treatment_samples, sample_list.loc[Level].unique()):
            group_labels.extend(len(s)*[group])
            samples[group] = s
        group_labels = np.array(group_labels)    
        
        temp_ion_df = temp_ion_df.drop(columns=['m/z', 'rt', 'FG', 'Size'])
        X = temp_ion_df.T  # Transpose if features are rows and samples are columns
    
        # Create binary labels for PLS-DA
        y = pd.get_dummies(group_labels).values
    
        # Perform PLS-DA
        scores, explained_variance = perform_pls_da(X, y)

        # Create the plot
        fig = go.Figure()
    
        # Define a color map for the groups
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'teal', 'lavender', 'olive', 'maroon', 'navy', 'turquoise', 'gold', 'lime', 'indigo', 'silver']  # Add more colors if you have more groups
        color_map = {label: color for label, color in zip(np.unique(group_labels), colors)}
        # Add traces for each group and calculate centroids and spreads
        centroids = {}
        for i, label in enumerate(np.unique(group_labels)):
            group_scores = scores[group_labels == label]
            centroid = np.mean(group_scores, axis=0)
            std_dev = np.std(group_scores, axis=0)
    
            # Store centroid and standard deviation for the circle
            centroids[label] = (centroid, std_dev)
            fig.add_trace(go.Scatter(
                x=group_scores[:, 0],  # First PLS component
                y=group_scores[:, 1],  # Second PLS component
                mode='markers',
                name=label,
                marker=dict(color=color_map[label]) , # Use color from the map
                text=samples[label],  # Hover text
                hoverinfo='text'
            ))
    
        # Add circles
        for label, (centroid, std_dev) in centroids.items():
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=centroid[0] - 2 * std_dev[0],  # Adjust the factor to change circle size
                y0=centroid[1] - 2 * std_dev[1],
                x1=centroid[0] + 2 * std_dev[0],
                y1=centroid[1] + 2 * std_dev[1],
                line_color=color_map[label],  # Match the circle color with the group
                line_width=2,
            )
    
        # Update layout
        fig.update_layout(
            xaxis_title=f'PLS Component 1 ({explained_variance[0]:.2f}% Variance)',
            yaxis_title=f'PLS Component 2 ({explained_variance[1]:.2f}% Variance)',
            template='plotly_white'
        )
        
        p_value = your_p_value_calculation_function(X, y)
        
        if Title == None:
            Title = 'No title'
        fig.update_layout(
        title={
            'text': f'PLS-DA - {Title} (p-value: {p_value:.3f})',
            'y': 0.95,  # Adjust this value as needed
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Conditions not respected<br>Need two parameters minimum (i.e treatment + control)",
            x=2,  # Center horizontally
            y=1.5,  # Center vertically
            showarrow=False,
            font=dict(
                size=14,  # Adjust the font size as needed
                color="red",  # Adjust the text color as needed
                family="Arial"  # Adjust the font family as needed
            ),
            align="center"  # Center-align the text
        )
        return fig


# feature group table module and statistic module
###############################################################################
# Function to create the column color formatting of feature group list
def column_colors():
    colors = px.colors.qualitative.Set1
    style_data_conditional = []
    treatment = 1
    for i, col in enumerate(fg_table_render.columns[1:]):
        if ' cor_p' in col: # first type of col in the order
            p_val_col = fg_table_render.columns[i+2]
            assumption_col = fg_table_render.columns[i+3]  # i + 2 becasue we sart the enumarate at 1: (see above)
            non_param_col = fg_table_render.columns[i+5] # cor p value col of non param test
            non_param_cor_col = fg_table_render.columns[i+4] # p value col of non param tes
            d = {'if':{'filter_query': f'{{{p_val_col}}} <= 0.05 && {{{assumption_col}}} = yes','column_id': p_val_col}, 'color': colors[2], 'fontWeight': 'bold'}
            style_data_conditional.append(d)
            d = {'if':{'filter_query': f'{{{p_val_col}}} <= 0.05 && {{{assumption_col}}} = yes','column_id': assumption_col}, 'color': colors[2], 'fontWeight': 'bold'}
            style_data_conditional.append(d)


            
            d = {'if':{'filter_query': f'{{{p_val_col}}} <= 0.05 && {{{assumption_col}}} = no','column_id': p_val_col}, 'color': colors[4], 'fontWeight': 'bold'}  
            style_data_conditional.append(d)
            d = {'if':{'filter_query': f'{{{p_val_col}}} <= 0.05 && {{{assumption_col}}} = no','column_id': assumption_col}, 'color': colors[4], 'fontWeight': 'bold'}  
            style_data_conditional.append(d)
            
            d = {'if':{'filter_query': f'{{{p_val_col}}} > 0.05 && {{{assumption_col}}} = yes','column_id': p_val_col}, 'color': colors[0], 'fontWeight': 'bold'} 
            style_data_conditional.append(d)
            d = {'if':{'filter_query': f'{{{p_val_col}}} > 0.05 && {{{assumption_col}}} = yes','column_id': assumption_col}, 'color': colors[0], 'fontWeight': 'bold'} 
            style_data_conditional.append(d)
            
            d = {'if':{'filter_query': f'{{{non_param_col}}} < 0.05 && {{{assumption_col}}} = no','column_id': non_param_col}, 'color': colors[1], 'fontWeight': 'bold'} 
            style_data_conditional.append(d)
            
            if treatment % 2 == 0:
                d = {'if':{'column_id': col}, 'backgroundColor':'rgb(240, 240, 240)'} 
                style_data_conditional.append(d)
                d = {'if':{'column_id': p_val_col}, 'backgroundColor':'rgb(240, 240, 240)'} 
                style_data_conditional.append(d)
                d = {'if':{'column_id': assumption_col}, 'backgroundColor':'rgb(240, 240, 240)'} 
                style_data_conditional.append(d)
                d = {'if':{'column_id': non_param_col}, 'backgroundColor':'rgb(240, 240, 240)'} 
                style_data_conditional.append(d)
                d = {'if':{'column_id': non_param_cor_col}, 'backgroundColor':'rgb(240, 240, 240)'} 
                style_data_conditional.append(d)
            
            treatment += 1
    return style_data_conditional
            
# Callback to generate 'plsda-plot' after 'network-graph' is rendered
@callback(
    Output('fg-table', 'data'),
    Output('fg-table', 'columns'),
    Output('fg-table', 'style_data_conditional'),
    Input('network-graph', 'figure'),
    prevent_initial_call = True
)
def create_fg_table(network_fig):
    columns = [{"name": i, "id": i} for i in fg_table_render.columns[1:]] # we drop first column which is sd_Table
    data = fg_table_render.iloc[:, 1:] # remove 'sd_table'
    data = data.to_dict('records') # create a list of dict, each dict is one row of the df
    style_data_conditional = column_colors()
    return data, columns, style_data_conditional
        
# Statistical tests and assumptions, Standardized is proposed as an option
def stat_n_assumptions(Component, Sample_list, Sample_presence_threshold = 100): # area as alternative
    """
    The sub table of a group of feature, corresponding to an ion, is taking accorrding to the features id in the component. A component is not more than
    a shared networks of nodes.
    Then the values are Standardizedd accross sampels, per feature. Standardized is chosen to take each time the mean as reference an not the highest values as other type of standardization.
    For each level of treatment (biotic or abiotic), the means of the samples for all features are taken, and used tomake the final means per treatment level.
    In the pre-processing, the number of sampels accorss the feature per treatment is taken to determinate if the number of smaples per treatment make sense for a statistical comparison. We cannot conapre 5 data points to 30 for example.
    Then minimal amount of sample should be present for the stat tests (min 3). and assumptions and the test are procceded according to the number of treatment present per level of treatment.
    """
    global featurelist, all_final_mean, all_sample_mean, project_loaded, preprocessed_df, meta_intensities, preprocessed_df_raw
    
    if project_loaded.meta:
        if meta_intensities:
            sd_table = pd.melt(preprocessed_df_raw, id_vars=['feature', 'm/z', 'rt'], var_name='sample', value_name='std_v')
        else:
            sd_table = pd.melt(preprocessed_df, id_vars=['feature', 'm/z', 'rt'], var_name='sample', value_name='std_v')
        sd_table = sd_table[sd_table['feature'].isin(Component)]  
    else:        
        table = featurelist[featurelist['feature'].isin(Component)]    
        columns_to_remove = ['peak_rt', 'peak_rt_start', 'peak_rt_end', 'peak_mz', 'peak_mz_min', 'peak_mz_max', 'MS_level', 'mzml_name']
        columns_to_drop = [col for col in columns_to_remove if col in table.columns]        
        table = table.drop(columns=columns_to_drop)        
        table.sort_values(by='m/z', ascending = False) # sort by descending ordering of mass
    
        # Standardize within each feature without relying on deprecated groupby.apply behaviour
        sd_table = standardize_group_Standardized(table, 'area')
        sd_table['std_v'] = sd_table['std_v'].round(2)
    
    all_sample_mean = {}
    all_final_mean = {}
    stat_result = {'Level':[], 'Test_type': [], 'Test_factor':[], 'Test_p-value':[], 'Assumption_validity' : [], 'Non_parametric_factor': [], 'Non_parametric_p-value': [], 'Normality_assumption': [], 'Equality_of_variance_assumption': [], 'Outliers_assumption': [], 'Post_Hoc_Results_tukey':[]}
    for level in Sample_list.index:  
        level_mapping = Sample_list.loc[level].to_dict() # dictionnary with sample as index and treatment as values for a specific treatement level
        sample_mean = sd_table.reset_index(drop=True).groupby('sample')['std_v'].mean() # here are the mean value per sample for all features in the feature group
        sample_mean = sample_mean.reset_index() # put sample as column and not index
        sample_mean[f'{level}_level_category'] = sample_mean['sample'].map(level_mapping) # apply level mapping to create treatment level column
        all_sample_mean[level] = sample_mean
        
        # have the average mean for each condition in the given level taking all sample mena
        final_mean = average_standardized_values(sd_table, Sample_list, level)
        final_mean['std_v'] = final_mean['std_v'].round(2)
        all_final_mean[level] = final_mean
        
        # Pre-processing to check if the number of sample per treatment is comparable
        level_nb = Sample_list.loc[level,:].unique() # it is the number of condition for this level of treament (e.g control vs treatment, or t1 vs t2 vs t3, etc) check the number of condition, if > 2 it will be anova test
        count_list = []
        sample_pres_threshold_validity = True
        for t in level_nb:
            nb_samples_per_level = len(sample_mean.loc[sample_mean[f'{level}_level_category'] ==  t, 'std_v']) # count the number of samples in each category 
            count_list.append(nb_samples_per_level)
            nb_samples_per_level_sup_0 = len(sample_mean.loc[(sample_mean[f'{level}_level_category'] == t) & (sample_mean['std_v'] > 0)])
            sample_pres_ratio = nb_samples_per_level_sup_0 / nb_samples_per_level
           
            if Sample_presence_threshold == 0:
                sample_pres_threshold_validity = True
            elif Sample_presence_threshold >= sample_pres_ratio:
                 sample_pres_threshold_validity = False
        #all_equal = max(count_list) - min(count_list) <= len(Component) # no more than the number of features in total amount of samples across all features as difference of data point between the treatments (this is a stat test assumption, almost equal nb of data point)
        no_below_three = all(x >= 3 for x in count_list) # min nb of samples for a treatment to make stat test is 3
        all_unique = (len(Sample_list.loc[level].unique()) == 1)
        
        # Proceed stat results
        if no_below_three and not all_unique and not sample_pres_threshold_validity: # if there are enough features in the feature group to make a test
            #stat_result['all_equal'] = all_equal
            stat_result['Level'].append(level)    
            # t test part
            if len(level_nb) == 2:
                stat_result['Test_type'].append('t-test')
                group1 = sample_mean.loc[sample_mean[f'{level}_level_category'] == level_nb[0], 'std_v']
                group2 = sample_mean.loc[sample_mean[f'{level}_level_category'] == level_nb[1], 'std_v']
                t_stat, p_val = stats.ttest_ind(group1, group2)
                if p_val <= 0.05:
                    valid = 'yes'
                else:
                    valid = 'no'
                stat_result['Test_factor'].append(format_number(t_stat))
                stat_result['Test_p-value'].append(format_number(p_val))

                # Testing for Normality
                if check_range(group1):
                    _, p1 = stats.shapiro(group1)
                else:
                    p1 = 'NaN' # Not applicable

                if check_range(group2):
                    _, p2 = stats.shapiro(group2)
                else:
                    p2 = 'NaN' # Not applicable
                if (p1 and p2 != 'Nan')  and  not(isinstance(p1, str)) and  not(isinstance(p2, str)):
                    stat_result['Normality_assumption'].append(f"Shapiro-Wilk Test P-Values: {level_nb[0]} group = {format_number(p1)}, {level_nb[1]} group = {format_number(p2)}\n")
                else:
                    stat_result['Normality_assumption'].append("Shapiro-Wilk Test non applicable")
                    p1 = 0
                    p2 = 0
                
                # Testing for Equality of Variances
                # Levene's Test
                _, p_var = stats.levene(group1, group2)
                stat_result['Equality_of_variance_assumption'].append(f"Levene's Test P-Value: {format_number(p_var)}\n")
                                    
                # Check for outliers
                outliers_group1 = abs(group1)[abs(group1) > 3]
                outliers_group2 = abs(group2)[abs(group2) > 3]
                stat_result['Outliers_assumption'].append(f"Outliers presence in {level_nb[0]} group = {len(outliers_group1)}, {level_nb[0]} group = {len(outliers_group2)}")
                
                # Check for all assumption validity
                if p_val <= 0.05 and p1 > 0.05 and p2 > 0.05 and p_var > 0.05 and (len(outliers_group1)+len(outliers_group2)) == 0:
                    stat_result['Assumption_validity'].append('yes')
                    stat_result['Non_parametric_factor'].append('-')
                    stat_result['Non_parametric_p-value'].append('-')
                else:
                    stat_result['Assumption_validity'].append('no')
                    statmann, pmann = stats.mannwhitneyu(group1, group2) # Mann-Whitney U Test (Wilcoxon Rank-Sum Test): Use this test instead of the t-test when comparing two independent groups. It's used when the data are not normally distributed.
                    stat_result['Non_parametric_factor'].append(format_number(statmann))
                    stat_result['Non_parametric_p-value'].append(format_number(pmann))
                
                stat_result['Post_Hoc_Results_tukey'].append('-')
                
            # ANOVA part                
            elif len(level_nb) > 2:
                stat_result['Test_type'].append('ANOVA')
                assumption_validity = True
                all_groups = []
                assumptions_details = "Shapiro-Wilk Test P-Values: "
                for treatment in level_nb:
                    
                    group = sample_mean.loc[sample_mean[f'{level}_level_category'] == treatment, 'std_v']
                    if check_range(group):
                        _, pshap = stats.shapiro(group)
                        assumptions_details += f"{treatment} group = {format_number(pshap)} " 
                        if pshap > 0.05:
                            assumption_validity = False
                    else:
                        pshap = 'NaN' # Not applicable
                        assumptions_details += f"{treatment} group = not applicable " 
                        assumption_validity = False
                    # _, pshap = stats.shapiro(group)
                    all_groups.append(group)

                stat_result['Normality_assumption'].append(assumptions_details)
                
                # Testing for Homogeneity of Variances
                if all(check_range(group) for group in all_groups):
                   _, p_var = stats.levene(*all_groups)
                else:
                   p_var = 'NaN' # Not applicable
                  
                if p_var != 'Nan' and  not(isinstance(p_var, str)):
                    stat_result['Equality_of_variance_assumption'].append(f"Levene's Test P-Value: {format_number(p_var)}\n")
                else:
                    stat_result['Equality_of_variance_assumption'].append("Levene's Test non applicable")
                    p_var = 0
                
                if p_var > 0.05: # the variance are not sig different from each other
                    assumption_validity = True
                else:
                    assumption_validity = False
                
                # Check for outliers
                outliers_details = "Outliers presence in "
                for treatment in level_nb:
                    group = sample_mean.loc[sample_mean[f'{level}_level_category'] == treatment, 'std_v']
                    outliers_group = abs(group)[abs(group) > 3]
                    outliers_details += f"{treatment} group = {len(outliers_group)}, "
                    if len(outliers_group) != 0:
                        assumption_validity = False
                stat_result['Outliers_assumption'].append(outliers_details[:-2]) # del the last ', '
                
                # Check for all assumption validity
                if assumption_validity:
                    stat_result['Assumption_validity'].append('yes')
                    stat_result['Non_parametric_factor'].append('-')
                    stat_result['Non_parametric_p-value'].append('-')
                else:
                    stat_result['Assumption_validity'].append('no')
                    statkruskal, pkruskal = stats.kruskal(*all_groups)
                    stat_result['Non_parametric_factor'].append(format_number(statkruskal))
                    stat_result['Non_parametric_p-value'].append(format_number(pkruskal))
                    
                # ANOVA test it 
                f_stat, p_value = stats.f_oneway(*all_groups)
                if p_value <= 0.05:
                    valid = 'yes'
                else:
                    valid = 'no'
                stat_result['Test_factor'].append(format_number(f_stat))
                stat_result['Test_p-value'].append(format_number(p_value))
                
                if p_value <= 0.05:
                    # Perform Tukey's HSD test for post hoc analysis
                    all_data = sample_mean['std_v'].values
                    all_groups_labels = sample_mean[f'{level}_level_category'].values
                    tukey_result = pairwise_tukeyhsd(endog=all_data, groups=all_groups_labels, alpha=0.05)
                    
                    tukey_summary = [] # Create a concise summary
                    for result in tukey_result.summary().data[1:]:  # Skip the header row
                        group1, group2, meandiff, p_adj, lower, upper, reject = result
                        if reject:
                            tukey_summary.append(f"{group1} vs {group2}: p={p_adj:.3f} (sig)")
                        else:
                            tukey_summary.append(f"{group1} vs {group2}: p={p_adj:.3f} (non-sig)")

                    stat_result['Post_Hoc_Results_tukey'].append("; ".join(tukey_summary))
                else:
                    stat_result['Post_Hoc_Results_tukey'].append('-')
                
            else: # if there is only one condition
                pass
    return sd_table, all_final_mean, all_sample_mean, pd.DataFrame(stat_result)

def check_range(group):
    return group.max() - group.min() > 0

# Function to have numbers well formated for the user
def format_number(number):
    if abs(number) < 0.01:
        # For small numbers, use scientific notation with two significant digits
        return "{:.2e}".format(number)
    else:
        # For larger numbers, round to two decimal places
        return "{:.2f}".format(number)
    
# Take the average of all samples standardized value for each label of the selected level
def average_standardized_values(Sd_table, Sample_list, Level):
    # Create a dictionary to map sample names to their level
    level_mapping = Sample_list.loc[Level].to_dict()

    # Add a new column to sd_table for level category
    Sd_table[f'{Level}_level_category'] = Sd_table['sample'].map(level_mapping)

    # Calculate the average of standardized heights for each level category
    final_means = Sd_table.groupby(f'{Level}_level_category')['std_v'].mean().reset_index() # replace Sd_table by filtered_data if you want to use it
    
    return final_means

# Function to make z score standardization per feature (group)
def standardize_group_Standardized(Group, Measure): # MEasure is height or area
    group_copy = Group.copy()

    if 'feature' in group_copy.columns:
        # Standardize values within each feature while preserving the original frame
        max_values = group_copy.groupby('feature')[Measure].transform('max')
        group_copy['std_v'] = group_copy[Measure] / max_values
    else:
        # Fallback for callers that provide a single feature subset
        max_value = group_copy[Measure].max()
        group_copy['std_v'] = group_copy[Measure] / max_value

    return group_copy

import collections
def _group_size_counts(graph):
    sizes = [len(c) for c in nx.connected_components(graph)]
    cnt = collections.Counter(sizes)  # {size: how_many_groups}
    return len(sizes), cnt


def _refine_component_by_shape(component_nodes, threshold, graph):
    if threshold <= 0 or not feature_shape_vectors:
        return [set(component_nodes)]
    nodes = set(component_nodes)
    project_name = _active_project_name()
    project_context = _active_project()
    logging_config.log_info(
        logger,
        "Shape refinement started for component (%s nodes, threshold=%.2f)",
        len(nodes),
        threshold,
        project=project_context,
    )
    available = []
    missing = []
    for node in nodes:
        vector = feature_shape_vectors.get(node)
        if vector is None:
            missing.append(node)
        else:
            available.append((node, vector))
    if len(available) < 2:
        logging_config.log_info(
            logger,
            "Shape refinement skipped for component (%s nodes) due to insufficient vectors",
            len(nodes),
            project=project_context,
        )
        return [set(nodes)]
    try:
        H = nx.Graph()
        H.add_nodes_from(node for node, _ in available)
        for idx, (node_i, vec_i) in enumerate(available):
            for jdx in range(idx + 1, len(available)):
                node_j, vec_j = available[jdx]
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i == 0 or norm_j == 0:
                    continue
                similarity = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                if similarity >= threshold:
                    H.add_edge(node_i, node_j, weight=similarity)
        shape_components = [set(comp) for comp in nx.connected_components(H)]
        if len(shape_components) <= 1:
            logging_config.log_info(
                logger,
                "Shape refinement produced a single subset for component (%s nodes)",
                len(nodes),
                project=project_context,
            )
            return [set(nodes)]
        for node in missing:
            neighbors = set(graph.neighbors(node)).intersection(nodes)
            if neighbors:
                best_idx = 0
                best_overlap = -1
                for idx, comp in enumerate(shape_components):
                    overlap = len(neighbors & comp)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = idx
                shape_components[best_idx].add(node)
            else:
                shape_components[0].add(node)
        logging_config.log_info(
            logger,
            "Shape refinement completed for component (%s nodes) with %s refined groups",
            len(nodes),
            len(shape_components),
            project=project_context,
        )
        return shape_components
    except Exception as exc:  # pragma: no cover - defensive logging
        logging_config.log_exception(
            logger,
            "Shape refinement failed for component (%s nodes)",
            len(nodes),
            project=project_context,
            exception=exc,
        )
        return [set(nodes)]

# Callback for updating the network graph
@callback(
    [Output('network-graph', 'figure'),
    Output('pca-level', 'options'),
    Output('volcano-level', 'options'),
    Output('plsda-level', 'options'),
    Output('pca-level', 'value'),
    Output('volcano-level', 'value'),
    Output('plsda-level', 'value')],
    [Input('update-button', 'n_clicks'),
     Input('network-intermediate', 'children'),
     Input('update-intensities', 'n_clicks')],
    State('rt-threshold', 'value'),
    State('correlation-threshold', 'value'), # it is also mz threshold in meta analysis
    State('shape-threshold', 'value'),
    prevent_initial_call = True
)
def update_graph(n_clicks, intermediate_signal, update_intensities_clicks, Rt_threshold, Correlation_threshold, Shape_threshold):
    global preprocessed_df, stand_preprocessed_df, preprocessed_df_raw
    global G, S, network_updated, updating, FG_pair_shared_features
    global stand_ion_df, ion_df, ion_df_raw
    global dropdown_items_binary, dropdown_items, fg_table, fg_table_render
    global validity, project_loaded, featurelist, meta_intensities
    global feature_shape_vectors
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if Shape_threshold is None:
        Shape_threshold = 0.0
    else:
        Shape_threshold = max(0.0, min(1.0, Shape_threshold))
    # Shape vectors remain available for chromatogram visualization, but
    # chromatographic similarity is no longer used to subdivide feature groups
    # during analytics grouping.
    shape_refinement_enabled = False

    if validity == False:
        raise PreventUpdate()
    if updating:
        raise PreventUpdate()
    """
    To make a network graph, we clear G, which is the object of network class.
    - similar_rt_groups takes all features grouped based on rt threshold
    - all_corr is a list created to take pair of features and corresponding spearmann correlation. Spearmann is used due to the non-parametric nature of the sample values.
        Basically it correlates a feature to another by making spearman correlation to all samples SHARED by the features.
    - edges is all_corr with > specific threshold, let's say 0.9. Attention is made to not have twice the same feature pair correlation with the line "feature < other_feature"
    - nodes of G are fed with the features. For the moment there is no network, only a cloud of dots (nodes)
    - edges are are added to the nodes, creating the groups representing similar features. Each group of feature a different potential precusor ion.
        Be careful, a in a group of feature with similar rt, some feature can correlate together and other don't, but some correlate in between.
        Indeed, if f1 coorelate to f2, f2 to f3, but f1 not to f3, there will be all together a feature group, but composed of two cliques
    - ion_df is a df with all unique precusor ion as rows, represented by average of rt and max m/z of the features which composed the precusor ion
    """
    
    if button_id == 'update-button' and not (project_loaded.meta == True):
        updating = True
        project_name = _active_project_name()
        project_context = _active_project()
        logging_config.log_info(
            logger,
            "Feature grouping update requested (RT=%.2f, Corr=%.2f, Shape=%.2f)",
            Rt_threshold,
            Correlation_threshold,
            Shape_threshold,
            project=project_context,
        )
        # Recreate the network graph based on new threshold values
        G.clear()  # Reset the graph
        sample_count = len(project_loaded.sample_names)
        if sample_count == 0:
            # Fall back to the intensity columns present on the preprocessed dataframe (m/z and rt are metadata)
            sample_count = max(0, len(preprocessed_df.columns) - 2)
        if sample_count == 0:
            logging.warning("No samples detected on project %s; disabling shared-coverage ratio threshold", project_loaded.project_name)
            minimum_ratio_threshold = 0.0
        else:
            minimum_ratio_threshold = 1 - (3 / sample_count)
            minimum_ratio_threshold = max(0.0, min(1.0, minimum_ratio_threshold))
        # 0.7 for 10 samples, it is nb of column (samples) with non-0 values at least to be shared between two features to validate a correlation. The more samples in an experiment, the higher this threshold to avoid chained wrong correlation
        
        with _feature_grouping_step_logger("RT neighborhood discovery"):
            similar_rt_groups = {}
            for feature in preprocessed_df.index:
                rt = preprocessed_df.loc[feature, 'rt']
                similar_features = preprocessed_df[np.abs(preprocessed_df['rt'] - rt) <= Rt_threshold].index.tolist()
                similar_rt_groups[feature] = similar_features

            unique_lists = list({tuple(v) for v in similar_rt_groups.values()})
            unique_lists = [list(t) for t in unique_lists]
            similar_rt_groups_adjusted = {}
            for feature in preprocessed_df.index:
                similar_rt_groups_adjusted[feature] = max((lst for lst in unique_lists if feature in lst), key=len, default=[])

            all_pairs_possible = [
                (feature, other_feature)
                for feature, group in similar_rt_groups_adjusted.items()
                for other_feature in group
                if feature < other_feature
            ]
            pairs_df = pd.DataFrame(all_pairs_possible)
            pairs_df.columns = ['feature', 'other_feature']
            rt_values = preprocessed_df.loc[:, 'rt']
            pairs_time_difference = np.abs(rt_values.values[:, None] - rt_values.values[None, :])
            pairs_time_difference = pd.DataFrame(pairs_time_difference, index=rt_values.index, columns=rt_values.index)
        
        with _feature_grouping_step_logger("Correlation and coverage computation"):
            valid_rows_df = preprocessed_df.drop(columns=['m/z', 'rt'])
            valid_rows_df = valid_rows_df.replace(0, np.nan)
            correlation_matrix_sp = valid_rows_df.transpose().corr(method='spearman').round(2)
            non_nan_mask = valid_rows_df.notna().astype(int)
            shared_non_nan = np.dot(non_nan_mask.values, non_nan_mask.values.T)
            total_non_nan = (non_nan_mask.values[:, None, :] | non_nan_mask.values[None, :, :]).sum(axis=2)
            ratio_matrix  = shared_non_nan / total_non_nan
            ratio_matrix = pd.DataFrame(ratio_matrix, index=valid_rows_df.index, columns=valid_rows_df.index)
        
        with _feature_grouping_step_logger("Edge filtering"):
            edges = []
            for _, row in pairs_df.iterrows():
                feature = row['feature']
                other_feature = row['other_feature']
                if feature in correlation_matrix_sp.index and other_feature in correlation_matrix_sp.columns:
                    corr_value = correlation_matrix_sp.loc[feature, other_feature]
                    ratio_value = ratio_matrix.loc[feature, other_feature]
                    rt_diff = pairs_time_difference.loc[feature, other_feature]
                    if (
                        corr_value >= Correlation_threshold
                        and ratio_value >= minimum_ratio_threshold
                        and rt_diff <= Rt_threshold
                    ):
                        edges.append(
                            (
                                feature,
                                other_feature,
                                corr_value,
                            )
                        )

        with _feature_grouping_step_logger("Graph construction and Louvain filtering"):
            for feature in preprocessed_df.index:
                G.add_node(feature, mz=preprocessed_df.loc[feature, 'm/z'])
            for node in G.nodes():
                G.nodes[node]['mz'] = preprocessed_df.loc[node, 'm/z']
                G.nodes[node]['rt'] = preprocessed_df.loc[node, 'rt']

            for feature, other_feature, weight in edges:
                G.add_edge(
                    feature,
                    other_feature,
                    weight=weight,
                    correlation=weight,
                )
            G_before = G.copy()
            total_before, counts_before = _group_size_counts(G_before)

            communities = nx_comm.louvain_communities(G, weight='weight', resolution= 1)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i

            edges_to_remove = []
            for u, v in G.edges():
                if community_map[u] != community_map[v]:
                    edges_to_remove.append((u, v))
            G.remove_edges_from(edges_to_remove)

            G_after = G.copy()
            G_after.remove_edges_from(edges_to_remove)
            total_after, counts_after = _group_size_counts(G_after)

            all_sizes = sorted(set(counts_before.keys()) | set(counts_after.keys()))
            row_before = {"state": "before", "total_groups": total_before}
            row_after  = {"state": "after",  "total_groups": total_after}

            for s in all_sizes:
                row_before[f"rank{s}"] = counts_before.get(s, 0)
                row_after[f"rank{s}"]  = counts_after.get(s, 0)

            df_summary = pd.DataFrame([row_before, row_after])
            rank_cols = [c for c in df_summary.columns if c.startswith("rank")]
            zero_both = [c for c in rank_cols if df_summary[c].sum() == 0]
            df_summary = df_summary.drop(columns=zero_both)
            out_csv = os.path.join(".", "group_size_summary.csv")
            df_summary.to_csv(out_csv, index=False)
            logging_config.log_info(
                logger,
                "Feature grouping Louvain summary saved to %s",
                out_csv,
                project=project_context,
            )

            isolated_nodes = [node for node, degree in G.degree() if degree == 0]

            ion_df = pd.DataFrame(columns = preprocessed_df.columns)
            stand_ion_df = pd.DataFrame(columns = preprocessed_df.columns)

        # Iterating through the list L and calculating the averages
        ID = 1 # correspond to group node (feature group) IDs
        for component in nx.connected_components(G): # component is a set of ions
            if shape_refinement_enabled:
                refined_components = _refine_component_by_shape(
                    component,
                    Shape_threshold,
                    G,
                )
            else:
                refined_components = [set(component)]
            for refined_nodes in refined_components:
                subset_df = preprocessed_df[preprocessed_df.index.isin(refined_nodes)]
                stand_subset_df = stand_preprocessed_df[stand_preprocessed_df.index.isin(refined_nodes)]
                if subset_df.empty or stand_subset_df.empty:
                    continue
                max_mass = subset_df['m/z'].max()
                mean_rt = subset_df['rt'].mean()
                subset_df = subset_df.drop(columns=['m/z', 'rt'])
                stand_subset_df = stand_subset_df.drop(columns=['m/z', 'rt'])

                # Calculate the average of the filtered rows (excluding the 'feature' column)
                avg_row = subset_df.mean(numeric_only=True).to_frame().transpose()
                stand_avg_row = stand_subset_df.mean(numeric_only=True).to_frame().transpose()

                avg_row['m/z'] = max_mass # highest mass is the reference mass, because it is more likely to be the closet to the precusor ion mass
                avg_row['rt'] = mean_rt
                avg_row['Size'] = len(subset_df) # number of features present
                avg_row['FG'] = ID
                stand_avg_row['m/z'] = max_mass # highest mass is the reference mass, because it is more likely to be the closet to the precusor ion mass
                stand_avg_row['rt'] = mean_rt
                stand_avg_row['Size'] = len(subset_df) # number of features present
                stand_avg_row['FG'] = ID

                # Now perform the concatenation
                ion_df = pd.concat([ion_df, avg_row], ignore_index=True)
                stand_ion_df = pd.concat([stand_ion_df, stand_avg_row], ignore_index=True)

                colors = px.colors.qualitative.Set1 # import colors (https://plotly.com/python/discrete-color/)
                for node in refined_nodes:
                    G.nodes[node]['feature_group'] = ID
                    G.nodes[node]['color'] = colors[8] # color is grey for the moment, because stats test are loading i background

                ID += 1 # feature group id
    
        # Convert the updated graph to a Plotly figure
        ion_df_raw = ion_df.copy(deep=True)
        updated_fig = convert_graph_to_2d_plotly_figure(G)  # Define this function to convert networkx graph to Plotly figure        
        network_updated = True        
        return updated_fig, dropdown_items, dropdown_items_binary, dropdown_items, dropdown_items[0]['value'], dropdown_items_binary[0]['value'], dropdown_items[0]['value']
    
    elif (button_id == 'update-button' or button_id == 'update-intensities') and project_loaded.meta == True:
        if button_id == 'update-intensities':
            meta_intensities = True
        else:
            meta_intensities = False

        experiments = project_loaded.experiments
        experiment_FG = project_loaded.experiment_FG
        feature_id = project_loaded.feature_id
        
        RT_threshold_meta = Rt_threshold
        MZ_threshold_meta = Correlation_threshold # called correlation threshold by defaut but in meta analysis it is MZ threshold for adducts matching
        
        potential_neutral_masses_groups = pd.DataFrame({'m/z_min':[], 'm/z_max':[], 'rt_min':[], 'rt_max':[], 'feature_id':[], 'FG_grouping':[], 'feature_pnm':[], 'adducts' : [], 'validity':[]})
        feature_id_pnm_matching = {key: [] for key in range(1, feature_id + 1)} # verifies if one feature's neutral masses matches several neutral mass groups, if yes, this has to be informed in the statistical results
        FG_rank_list = []
        for exp_title in experiments.keys():
            # pnm grouping part
            for fg in experiments[exp_title]['FG']:
                size = len(experiments[exp_title]['FG'][fg]['absolute_subset_df'])
                if size > 0: # can be filter if there is at least 2 features in the FG (to have minimal interest) with > 1
                    new_row_fg = {
                            'experiment': exp_title,
                            'FG': fg,
                            'Size': size}
                    FG_rank_list.append(new_row_fg)
                    pnm = experiments[exp_title]['FG'][fg]['neutral_masses_possibilities_dict'] # dict of all neutral mass possibilities from this fg
                    for neutral_mass in pnm.keys():
                        rt = round(pnm[neutral_mass][4], 2) # take the feature rt
                        nm = round(neutral_mass, 3)
                        fg = pnm[neutral_mass][1]
                        condition = (potential_neutral_masses_groups['m/z_min'] <= nm) & (potential_neutral_masses_groups['m/z_max'] >= nm) & (potential_neutral_masses_groups['rt_min'] <= rt) & (potential_neutral_masses_groups['rt_max'] >= rt)
                        
                        result = potential_neutral_masses_groups.loc[condition]                
                        if result.empty:                        
                            index = len(potential_neutral_masses_groups)
                            feature_id_pnm_matching[pnm[neutral_mass][2]].append(index)
                            new_row = pd.DataFrame({'m/z_min': nm - MZ_threshold_meta, 
                                                    'm/z_max': nm + MZ_threshold_meta, 
                                                    'rt_min': rt - RT_threshold_meta, 
                                                    'rt_max': rt + RT_threshold_meta, 
                                                    'feature_id' : [[pnm[neutral_mass][2]]], 
                                                    'FG_grouping': [[pnm[neutral_mass][1]]], 
                                                    'feature_pnm': [[pnm[neutral_mass]]],
                                                    'adducts' : [[pnm[neutral_mass][5]]],
                                                    'validity': False}) # validity is True if at least two different adducts are dected for the potential neutral mass
                            potential_neutral_masses_groups = pd.concat([potential_neutral_masses_groups, new_row], ignore_index=True)
                        else:
                            FG_association = check_FG(experiment_FG, fg, result['FG_grouping'])
                            if FG_association:
                                index = potential_neutral_masses_groups[condition].index[0]
                                feature_id_pnm_matching[pnm[neutral_mass][2]].append(index)
                                if rt < result['rt_min'].values[0]:
                                    new_rt_min = result['rt_min'].values[0] - RT_threshold_meta
                                    new_rt_max = result['rt_max'].values[0]
                                elif rt > result['rt_max'].values[0]:
                                    new_rt_min = result['rt_min'].values[0] 
                                    new_rt_max = result['rt_max'].values[0] + RT_threshold_meta
                                else:
                                    new_rt_min = result['rt_min'].values[0] 
                                    new_rt_max = result['rt_max'].values[0]
                                    
                                if nm < result['m/z_min'].values[0]:
                                    new_mz_min = result['m/z_min'].values[0] - MZ_threshold_meta
                                    new_mz_max = result['m/z_max'].values[0]
                                elif nm > result['m/z_max'].values[0]:
                                    new_mz_min = result['m/z_min'].values[0] 
                                    new_mz_max = result['m/z_max'].values[0] + MZ_threshold_meta
                                else:
                                    new_mz_min = result['m/z_min'].values[0] 
                                    new_mz_max = result['m/z_max'].values[0]
                                potential_neutral_masses_groups.loc[index, ['rt_min', 'rt_max']] = [new_rt_min, new_rt_max]
                                potential_neutral_masses_groups.loc[index, ['m/z_min', 'm/z_max']] = [new_mz_min, new_mz_max]
                                potential_neutral_masses_groups.at[index, 'feature_id'].append(pnm[neutral_mass][2])
                                potential_neutral_masses_groups.at[index, 'FG_grouping'].append(pnm[neutral_mass][1])
                                potential_neutral_masses_groups.at[index, 'feature_pnm'].append(pnm[neutral_mass])
                                potential_neutral_masses_groups.at[index, 'adducts'].append(pnm[neutral_mass][5])
                                all_adducts = [item for sublist in result['adducts'] for item in sublist]
                                unique_adducts = set(all_adducts)
                                if len(unique_adducts) > 1:
                                    potential_neutral_masses_groups.at[index, 'validity'] = True
        #potential_neutral_masses_groups = potential_neutral_masses_groups[potential_neutral_masses_groups['feature_id'].apply(len) != 1] # filter out all the freatures with low lenghts
        potential_neutral_masses_groups = potential_neutral_masses_groups[potential_neutral_masses_groups['validity'] == True] # remove all rows wehre there is only a similar mass association, because for meta analsis we rely only on neutral masses
        
        FG_grouping = potential_neutral_masses_groups['FG_grouping'].to_list()
        FG_grouping_sorted_lists = [sorted(lst) for lst in FG_grouping]
        FG_grouping_unique_lists = {tuple(lst) for lst in FG_grouping_sorted_lists if len(lst) > 1} # Filter unique lists with length greater than 1 (meaning when there are two feature groups associated because they have at least one same potential neutral mass between them)
        FG_grouping_unique_lists = [list(lst) for lst in FG_grouping_unique_lists]
        
        FGs = []
        edges = {}
        FG_pair_shared_features = {}
        unique_pairs = []
        
        for ix, row in potential_neutral_masses_groups.iterrows():
            pairs_f_id = list(itertools.combinations(row['feature_id'], 2))
            pairs_fg = list(itertools.combinations(row['FG_grouping'], 2))
            equal = row['validity'] # to know if the m/z are the same or it is an inferred neutral mass
            if equal:
                w = 0.5  # if it is inferred mass, there is less confidency, we add only 0.5
            else:
                w = 1.0 # if it is the same mass, a weight of 1 is added to the FG/FG connection
            for p_id, p_fg in zip(pairs_f_id, pairs_fg):
                if p_id not in unique_pairs:
                    unique_pairs.append(p_id)
                    if p_fg[0] != p_fg[1]:
                        if p_fg not in edges.keys():
                            edges[p_fg] = w
                            FG_pair_shared_features[p_fg] = [ix]
                        else:
                            edges[p_fg] += w
                            FG_pair_shared_features[p_fg].append(ix)
                        if p_fg[0] not in FGs:
                            FGs.append(p_fg[0])
                        if p_fg[1] not in FGs:
                            FGs.append(p_fg[1])

        for fg in FGs:
            S.add_node(fg)
        
        experiment_index = {exp : c for c, exp in enumerate(experiments.keys())}
        experiment_combinations = list(itertools.combinations(list(experiment_index.values()), 2))
        values = list(range(1, len(experiment_combinations) + 1)) # Assign values from 1 to all possible combination of experiments
        
        # Create a list of all pairs including opposite pairs with corresponding values
        all_pairs = []
        for pair, value in zip(experiment_combinations, values):
            all_pairs.append((pair, value))
            all_pairs.append(((pair[1], pair[0]), value))
        
        # Create DataFrame
        all_pairs = pd.DataFrame(all_pairs, columns=['combination', 'value'])
        
        for node in S.nodes():
            exp = experiment_FG[node]['experiment']        
            S.nodes[node]['exp'] = experiment_index[exp] # experiment_index[exp] # add experiment title corresponding number

            
        for p_fg, w in edges.items():
            S.add_edge(p_fg[0], p_fg[1], weight=w)
            
        # Identify connected components and process each
        components = list(nx.connected_components(S))
        for component in components:
            edges_to_remove = process_component(component, S, experiment_index, all_pairs)
            
        S.remove_edges_from(edges_to_remove)
        isolated_nodes = list(nx.isolates(S)) # Step 3: Identify isolated nodes (no meta feature group, only single fg)
        #S.remove_nodes_from(isolated_nodes) # Remove isolated nodes, meaninf filter out the feature groups with a size of 1

        components = list(nx.connected_components(S))
        MFG = {'mfg_id':[], 'FGs':[], 'pnm_index':[]}
        for count, component in enumerate(components):
            MFG['mfg_id'].append(count+1)
            index_to_append = []
            fg_to_append = []
            for c in component:
                fg_to_append.append(c)
                for pair, pnm_indexes in FG_pair_shared_features.items():
                    if c in pair:
                        index_to_append.extend(pnm_indexes)
            index_to_append = list(set(index_to_append))
            MFG['FGs'].append(fg_to_append)
            MFG['pnm_index'].append(index_to_append)
        
        from itertools import chain
        pnm_list = list(chain.from_iterable(MFG['pnm_index']))
        
        preprocessed_df = {'feature':[p for p in range(len(pnm_list))], 'pnm_index':[p for p in pnm_list], 'm/z':[0.0 for p in range(len(pnm_list))], 'rt':[0.0 for p in range(len(pnm_list))]}
        samples = project_loaded.msn_df_deblanked['sample'].unique()
        for sample in samples:
            preprocessed_df[sample] = [0 for p in range(len(pnm_list))]
        preprocessed_df = pd.DataFrame(preprocessed_df)
        preprocessed_df_raw = preprocessed_df.copy(deep=True)
        
        adduct_df = {'feature':[p for p in range(len(pnm_list))], 'pnm_index':[p for p in pnm_list]} # adduct_df is here to indicate which feature is detected per sample when there is a pnm
        samples =  project_loaded.msn_df_deblanked['sample'].unique()
        for sample in samples:
            adduct_df[sample] = ['' for p in range(len(pnm_list))]
        adduct_df = pd.DataFrame(adduct_df)
            
        for sample in project_loaded.msn_df_deblanked['sample'].unique():
            for t in project_loaded.tables_list:
                if sample in t['samples']:
                    exp_title = t['exp_title']
                    esi_mode = project_loaded.template_esi_mode[exp_title]
                    if esi_mode == 'pos':
                        adducts = pd.read_csv('./data/positive_mode_adducts.csv')
                    elif esi_mode == 'neg':
                        adducts = pd.read_csv('./data/negative_mode_adducts.csv')
                    break
            
            spectra = project_loaded.files_spectra[sample]
            for feature, pnm in enumerate(pnm_list):
                rt_min = potential_neutral_masses_groups.loc[pnm, 'rt_min']
                rt_max = potential_neutral_masses_groups.loc[pnm, 'rt_max']
                rt = round((rt_min + rt_max) / 2, 2)
                #if neutral:
                mz_min = potential_neutral_masses_groups.loc[pnm, 'm/z_min']
                mz_max = potential_neutral_masses_groups.loc[pnm, 'm/z_max']  
                mz = round((mz_min + mz_max) / 2, 3)
                p_mz_min = [mz_min - m_c for m_c in adducts['mass_change']]
                p_mz_max = [mz_max - m_c for m_c in adducts['mass_change']]

                for mz_min, mz_max, add in zip(p_mz_min, p_mz_max, adducts['adduct']):
                    for rt_ms1, rt_ms2 in zip(spectra[0].keys(), spectra[1].keys()):
                        if rt_ms1 >= rt_min and rt_ms1 <= rt_max:
                            array = spectra[0][rt_ms1]
                            result = (array[:, 0] >= mz_min) & (array[:, 0] <= mz_max)                            
                            if result.any():
                                matching_pairs = array[result] # Extract the pairs of values from the array where the condition is met
                                intensities = matching_pairs[:, 1] # Extract the corresponding intensity values from column 1
                                intensities = np.mean(intensities)
                                adduct_df.loc[feature, sample] = add # add to adduct df
                                break
                        if rt_ms2 >= rt_min and rt_ms2 <= rt_max:
                            array = spectra[1][rt_ms2]
                            result = (array[:, 0] >= mz_min) & (array[:, 0] <= mz_max)
                            if result.any():
                                matching_pairs = array[result] # Extract the pairs of values from the array where the condition is met
                                intensities = matching_pairs[:, 1] # Extract the corresponding intensity values from column 1
                                intensities = np.mean(intensities)
                                adduct_df.loc[feature, sample] = add # add to adduct df
                                break

                    if result.any():
                        break
                if result.any(): # if a mass is detected at ms1 or ms2, whatever the intensity
                    preprocessed_df.loc[feature, sample] = 1
                    preprocessed_df_raw.loc[feature, sample] = int(intensities) # be careful, the data in preprocessed_df is 0, so they are int() so you must add a int type, cannot addd a float type
                    
                if preprocessed_df.loc[feature, 'rt'] == 0:
                    preprocessed_df.loc[feature, 'rt'] = rt
                    preprocessed_df_raw.loc[feature, 'rt'] = rt
                if preprocessed_df.loc[feature, 'm/z'] == 0:
                    preprocessed_df.loc[feature, 'm/z'] = mz
                    preprocessed_df_raw.loc[feature, 'm/z'] = mz

        preprocessed_df.index = preprocessed_df['pnm_index']
        preprocessed_df = preprocessed_df.drop(columns=['pnm_index'])
        preprocessed_df = preprocessed_df[~preprocessed_df.index.duplicated(keep='first')]
        
        preprocessed_df_raw.index = preprocessed_df_raw['pnm_index']
        preprocessed_df_raw = preprocessed_df_raw.drop(columns=['pnm_index'])
        preprocessed_df_raw = preprocessed_df_raw[~preprocessed_df_raw.index.duplicated(keep='first')]
        
        # Make the network
        G.clear()
        # Create the edges list
        edges = [] # edges list tuples of len 3, with two correlated feature IDs and their corresponding correlation
        # Iterate over the pairs in pairs_df
        for pnms in MFG['pnm_index']:
            pnm_combinations = list(itertools.combinations(pnms, 2))
            for pair in pnm_combinations:
                f1_index = int(preprocessed_df.loc[pair[0], 'feature'])
                f2_index = int(preprocessed_df.loc[pair[1], 'feature'])
                edges.append((f1_index, f2_index, 1)) # 1 correspond to the association which is 

        for feature in preprocessed_df['feature']:
            G.add_node(feature, mz=preprocessed_df.loc[preprocessed_df['feature'] == feature, 'm/z'].values[0])
        for node in G.nodes():
            G.nodes[node]['mz'] = preprocessed_df.loc[preprocessed_df['feature'] == node, 'm/z'].values[0]
            G.nodes[node]['rt'] = preprocessed_df.loc[preprocessed_df['feature'] == node, 'rt'].values[0]
        G.add_weighted_edges_from(edges) # Add edges with correlation as edge attribute
        
        # Prepare ion_df
        ion_df = pd.DataFrame(columns = preprocessed_df_raw.columns) # ion_df will be all the unique precusor ion, with highest m/z as default representing m/z. Each line of ion_df represent a precusor ion, which is composed of one or multiple features
        ion_df = ion_df.drop(columns=['feature'])  
        stand_ion_df = ion_df.copy(deep=True)
        for mfg_index, pnms, component in zip(MFG['mfg_id'], MFG['pnm_index'], nx.connected_components(G)): # component is a set of ions   
            # Filter the rows in p_df where feature is in the current set
            stand_subset_df = preprocessed_df[preprocessed_df.index.isin(pnms)]            
            subset_df = preprocessed_df_raw[preprocessed_df_raw.index.isin(pnms)]            
            max_mass = subset_df['m/z'].max()
            mean_rt = subset_df['rt'].mean()
            subset_df = subset_df.drop(columns=['m/z', 'rt', 'feature'])
            stand_subset_df = stand_subset_df.drop(columns=['m/z', 'rt', 'feature'])
        
            # Calculate the average of the filtered rows (excluding the 'feature' column)
            stand_avg_row = stand_subset_df.mean(numeric_only=True).to_frame().transpose()    
            avg_row = subset_df.mean(numeric_only=True).to_frame().transpose()    
            
            avg_row['m/z'] = max_mass # highest mass is the reference mass, because it is more likely to be the closet to the precusor ion mass
            avg_row['rt'] = mean_rt
            avg_row['Size'] = len(subset_df) # number of features present in the MFG
            avg_row['FG'] = mfg_index
            stand_avg_row['m/z'] = max_mass # highest mass is the reference mass, because it is more likely to be the closet to the precusor ion mass
            stand_avg_row['rt'] = mean_rt
            stand_avg_row['Size'] = len(subset_df) # number of features present
            stand_avg_row['FG'] = mfg_index
            
            # Now perform the concatenation
            ion_df = pd.concat([ion_df, avg_row], ignore_index=True)
            stand_ion_df = pd.concat([stand_ion_df, stand_avg_row], ignore_index=True)
    
            colors = px.colors.qualitative.Set1 # import colors (https://plotly.com/python/discrete-color/)
            for node in component:
                G.nodes[node]['feature_group'] = mfg_index
                G.nodes[node]['color'] = colors[8] # color is grey for the moment, because stats test are loading in background
        
        ion_df_raw = ion_df.copy(deep=True)
        updated_fig = convert_graph_to_2d_plotly_figure(G)
        network_updated = True
        logging_config.log_info(
            logger,
            "Feature grouping update completed successfully (%s groups)",
            ID - 1,
            project=project_context,
        )
        return updated_fig, dropdown_items, dropdown_items_binary, dropdown_items, dropdown_items[0]['value'], dropdown_items_binary[0]['value'], dropdown_items[0]['value']
    
    elif button_id == 'network-intermediate':
        if intermediate_signal == 'Done':
            updated_fig = convert_graph_to_2d_plotly_figure(G)
            return updated_fig, dropdown_items, dropdown_items_binary, dropdown_items, dropdown_items[0]['value'], dropdown_items_binary[0]['value'], dropdown_items[0]['value']
        else:
            raise PreventUpdate()
    else:
        raise PreventUpdate()


# Function to process and clean each component
def process_component(component, graph, experiment_index, all_pairs):
    global S
    edges_to_remove = []
    subgraph = graph.subgraph(component).copy()    
    edges_data = [(u, v, data['weight'], 0) for u, v, data in subgraph.edges(data=True)]
    edges_data = pd.DataFrame(edges_data, columns=['fg1', 'fg2', 'weight', 'edge_nature']) # edge naure correspond to the type of pairwise association of experiment between the two FG from an edge
    edges_data = edges_data.sort_values(by='weight', ascending=False) # sort based on descending order
    
    for _, row in edges_data.iterrows():
        fg1 = int(row['fg1'])
        fg2 = int(row['fg2'])
        exp1 = S.nodes[fg1]['exp']
        exp2 = S.nodes[fg2]['exp']
        value = all_pairs.loc[all_pairs['combination'] == (exp1, exp2), 'value'].values[0]
        edges_data.loc[_, 'edge_nature'] = value
        
        
    # Reinitialize data structures to track components and their properties
    components = []  # List of components, each component is a set of nodes
    components_weights = []
    component_edges = []  # List of edge natures for each component
    # Iterate through sorted edges and build the network
    for _, row in edges_data.iterrows():
        fg1, fg2, weight, edge_nature = row['fg1'], row['fg2'], row['weight'], row['edge_nature']
        component_idx = find_or_create_component(fg1, fg2, edge_nature, components, component_edges, components_weights, experiment_index)
        components[component_idx].update([fg1, fg2])
        components_weights[component_idx] += weight
        component_edges[component_idx].add(edge_nature)
    
    components_dict = {}
    count = 0
    for comp, weight in zip(components, components_weights):
        components_dict[count] = (comp, weight)
        count += 1
        
    to_delete = []
    for ix, v  in components_dict.items():
        if ix not in to_delete:
            comp = v[0]
            weight = v[1]
            for ix2, v2 in components_dict.items():
                if ix2 not in to_delete and ix != ix2:
                    other_comp = v2[0]
                    other_weight = v2[1]
                    intersection = comp.intersection(other_comp)
                    if intersection:
                        if len(comp) > len(other_comp):
                            to_delete.append(ix2)
                        elif len(comp) == len(other_comp):
                            if weight > other_weight:
                                to_delete.append(ix2)
                            elif weight == other_weight:
                                to_delete.append(ix2) # because conditions are exactly the same we decide to simply supress one of the two components
                            else:
                                to_delete.append(ix)
                        else:
                            to_delete.append(ix) # dont test the weight, higher conneciton between different experiments is more important a MFG than the number of potential neutral masses
            
    for key in to_delete:
        if key in components_dict:
            del components_dict[key]
            
    for _, row in edges_data.iterrows():
        fg1 = row['fg1']
        fg2 = row['fg2']
        for comp, w in components_dict.values():
            if fg1 in comp and fg2 not in comp:
                edges_to_remove.append((fg1, fg2))
            elif fg2 in comp and fg1 not in comp:
                edges_to_remove.append((fg1, fg2))
                
    return edges_to_remove



@callback(
    [Output('network-intermediate', 'children'),
     Output('network-interval', 'disabled'),
     Output('fg-table-progress', 'value'),
     Output("fg-table-progress", "label"),
     Output('fg-table-progress-div', 'style'),
     Output('fg-table-div', 'style'),
     Output('volcano-table-progress', 'value'),
     Output("volcano-table-progress", "label"),
     Output('volcano-table-progress-div', 'style'),
     Output('volcano-table-div', 'style')],
    [Input('update-button', 'n_clicks'),
     Input('network-interval', 'n_intervals')],
    State('sample-threshold', 'value'),
    prevent_initial_call = True
)
def statistical_run(n_clicks, n, sample_threshold):
    global fg_table, fg_table_render, stat_proces_thread, network_updated, updating, loading_progress, pres_sample_threshold
    pres_sample_threshold = sample_threshold
    if network_updated and stat_proces_thread == None:
        logging_config.log_info(
            logger,
            "Statistical tests queued after feature grouping",
            project=_active_project(),
        )
        stat_proces_thread = start_run()
        return dash.no_update, None, 0, '0 %', dash.no_update, dash.no_update, 0, '0 %', dash.no_update, dash.no_update
    elif network_updated and stat_proces_thread.is_alive():
        return dash.no_update, None, loading_progress, f'{loading_progress} %', dash.no_update, dash.no_update, loading_progress, f'{loading_progress} %', dash.no_update, dash.no_update
    elif network_updated and not(stat_proces_thread.is_alive()):
        logging_config.log_info(
            logger,
            "Statistical tests finished",
            project=_active_project(),
        )
        stat_proces_thread = None
        network_updated = False
        updating = False
        return 'Done', True, 100, '100 %',{'display': 'none'}, {'maxWidth': '100%', 'overflowX': 'auto', 'overflow-y': 'hidden'}, 100, '100 %',{'display': 'none'}, {'maxWidth': '100%', 'overflowX': 'hidden'}
    return dash.no_update, None, 0, 100 ,dash.no_update, dash.no_update, 0, 100 ,dash.no_update, dash.no_update

# This method starts the run method in a new thread
def start_run():
    logging_config.log_info(
        logger,
        "Launching statistical test thread",
        project=_active_project(),
    )
    thread = threading.Thread(target=generate_stat)
    thread.start()
    return thread

# Function for Benjamini-Hochberg correction, i.e adjusting p values when there is a important number of features, to make the p values more robust and reduce false positive
def correction(fg_table):
    for c in fg_table.columns:
        if c[-1] == 'p':  # Check if column name ends with 'p', indicating a p-value column
            filtered_df = fg_table.copy(deep=True)

            # Step 1: Replace '-' values with NaN
            filtered_df[c].replace('-', np.nan, inplace=True)

            # Step 2: Convert p_value column to numeric
            filtered_df['corrected'] = filtered_df[c].astype(float)

            # Step 3: Drop NaN values from the corrected column
            p_values = filtered_df['corrected'].dropna().values

            # Step 4: Check if there are any valid p-values to correct
            if len(p_values) > 0:
                # Step 5: Apply Benjamini-Hochberg correction
                corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

                # Step 6: Assign the corrected p-values back to the DataFrame, ensuring the correct alignment
                filtered_df.loc[filtered_df['corrected'].notna(), 'corrected_p_value'] = corrected_p_values

                # Step 7: Add the corrected p-values as a new column
                new_col = c[:-1] + 'cor_p'  # for corrected_p value
                position = fg_table.columns.get_loc(c)  # Find the index where 'c' is located and insert before it
                fg_table.insert(position, new_col, filtered_df['corrected_p_value'].round(3))  # Insert the new column
            else:
                print(f"No valid p-values found in column {c}. Skipping correction.")
                
    return fg_table

# Funciton to realize statistical test or all treatment classes
def generate_stat():
    global fg_table, fg_table_render, G, sample_list, project_loaded, meta_intensities, preprocessed_df_raw, preprocessed_df, pres_sample_threshold

    project_name = _active_project_name()
    project_context = _active_project()
    logging_config.log_info(
        logger,
        "Statistical tests started",
        project=project_context,
    )
    try:
        fg_table = {'sd_table':[], 'm/z (Da)':[], 'rt (min)':[], 'FG':[], 'Size':[], 'Nodes' : []}
        test_type = {}
        for level in sample_list.index:
            test_type[level] = 'no_test'
            fg_table[level+' no_test p'] = []
            fg_table[level+' no_test a'] = []
            fg_table[level+' no_test np-p'] = []

        colors = px.colors.qualitative.Set1 # import colors (https://plotly.com/python/discrete-color/)
        FG = 1 # correspond to group node (feature group) IDs
        components = list(nx.connected_components(G))
        num_components = len(components)
        for c, component in enumerate(components):
            global loading_progress
            loading_progress = int((c/num_components)*100)
            if project_loaded.meta:
                if meta_intensities:
                    subset_df = preprocessed_df_raw.loc[preprocessed_df_raw['feature'].isin(component)]
                else:
                    subset_df = preprocessed_df.loc[preprocessed_df['feature'].isin(component)]
            else:
                subset_df = preprocessed_df[preprocessed_df.index.isin(component)]
            max_mass = subset_df['m/z'].max()
            mean_rt = subset_df['rt'].mean()
            subset_df = subset_df.drop(columns=['m/z', 'rt'])
            if project_loaded.meta:
                subset_df = subset_df.drop(columns=['feature'])

            sd_table, all_final_means, all_sample_mean, stat_result = stat_n_assumptions(component, sample_list, pres_sample_threshold)
            fg_table['m/z (Da)'].append(max_mass)
            fg_table['rt (min)'].append(round(mean_rt,2))
            fg_table['FG'].append(FG)
            fg_table['Size'].append(len(subset_df))
            fg_table['Nodes'].append(component)

            for level in sample_list.index:
                if stat_result.loc[stat_result['Level'] == level,].any().any():
                    test = stat_result.loc[stat_result['Level'] == level, 'Test_type'].values[0]
                    test_result = float(stat_result.loc[stat_result['Level'] == level, 'Test_p-value'].values[0])
                    assumpion_validity = stat_result.loc[stat_result['Level'] == level, 'Assumption_validity'].values[0]
                    non_param_pvalue = stat_result.loc[stat_result['Level'] == level, 'Non_parametric_p-value'].values[0]
                    test_type[level] = test
                    if level+f' {test} p' not in fg_table.keys():
                        existing_values = fg_table[level+' no_test p']
                        fg_table[level+f' {test} p'] = existing_values
                        fg_table[level+f' {test} p'].append(test_result)
                        fg_table.pop(level+' no_test p', None)

                        existing_values = fg_table[level+' no_test a']
                        fg_table[level+f' {test} a'] = existing_values
                        fg_table[level+f' {test} a'].append(assumpion_validity)
                        fg_table.pop(level+' no_test a', None)

                        existing_values = fg_table[level+' no_test np-p']
                        fg_table[level+f' {test} np-p'] = existing_values
                        fg_table[level+f' {test} np-p'].append(non_param_pvalue)
                        fg_table.pop(level+' no_test np-p', None)
                    else:
                        fg_table[level+f' {test} p'].append(test_result)
                        fg_table[level+f' {test} a'].append(assumpion_validity)
                        fg_table[level+f' {test} np-p'].append(non_param_pvalue)
                else:
                    test = test_type[level]
                    fg_table[level+f' {test} p'].append('-')
                    fg_table[level+f' {test} a'].append('-')
                    fg_table[level+f' {test} np-p'].append('-')

            for node in component:
                G.nodes[node]['FG'] = FG
                G.nodes[node]['color'] = colors[8]

            FG += 1
            fg_table['sd_table'].append(sd_table)

        fg_table = pd.DataFrame(fg_table)
        fg_table = correction(fg_table)

        for ix, row in fg_table.iterrows():
            color_pattern = []
            for level in sample_list.index:
                t_type = test_type[level]
                res = row[f'{level} {t_type} p']
                if res != '-':
                    test_result = float(row[f'{level} {t_type} p'])
                    assumpion_validity = row[f'{level} {t_type} a']
                else:
                    test_result = 1
                    assumpion_validity = 'no'
                    non_param_pvalue = 1
                FG = row['FG']
                if test_result <= 0.05 and assumpion_validity == 'yes':
                    color_pattern.append('Green')
                elif test_result <= 0.05 and assumpion_validity == 'no':
                    color_pattern.append('Orange')
                elif test_result >= 0.05 and assumpion_validity == 'yes':
                    color_pattern.append('Red')
                elif test_result >= 0.05 and assumpion_validity == 'no' and float(non_param_pvalue) <= 0.05:
                    color_pattern.append('Blue')
                else:
                    color_pattern.append('Grey')
            for node in row['Nodes']:
                if 'Green' in color_pattern:
                    G.nodes[node]['color'] = colors[2]
                elif 'Orange' in color_pattern:
                    G.nodes[node]['color'] = colors[4]
                elif 'Red' in color_pattern:
                    G.nodes[node]['color'] = colors[0]
                elif 'Blue' in color_pattern:
                    G.nodes[node]['color'] = colors[1]
                else:
                    G.nodes[node]['color'] = colors[8]
        fg_table = fg_table.drop(columns = ['Nodes'])
        fg_table_render = fg_table.copy(deep=True)
        logging_config.log_info(
            logger,
            "Statistical tests completed for %s feature groups",
            len(fg_table['FG']),
            project=project_context,
        )
        return 'Done'
    except Exception as exc:  # pragma: no cover - defensive logging
        logging_config.log_exception(
            logger,
            "Statistical tests failed",
            project=project_context,
            exception=exc,
        )
        raise



def convert_graph_to_2d_plotly_figure(Network):
    global network_fig, ion_df, preprocessed_df, project_loaded
    # Use a 2D layout
    pos_2d = nx.spring_layout(Network, dim=2, seed=42)
    
    # Nodes
    node_x = [pos_2d[node][0] for node in Network.nodes()]
    node_y = [pos_2d[node][1] for node in Network.nodes()]
    
    node_color_map = [Network.nodes[node]['color'] for node in Network.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers',
        marker=dict(
            size=10,
            color=node_color_map,
            opacity=0.8
        ),
        text=[f"m/z: {G.nodes[node]['mz']} RT:{G.nodes[node]['rt']} FG/MFG:{G.nodes[node]['feature_group']} feature:{node}" for node in G.nodes()],
        hoverinfo='text',
        showlegend=False  # Exclude edges from the legend
    )
    
    # Edges
    edge_x = []
    edge_y = []
    for edge in Network.edges():
        x0, y0 = pos_2d[edge[0]]
        x1, y1 = pos_2d[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y, 
        mode='lines',
        line=dict(width=0.5, color='black'),
        hoverinfo='none',
        showlegend=False  # Exclude edges from the legend
    )
    
    # Add legend items
    colors = px.colors.qualitative.Set1
    legend_items = {
        'Green': [colors[2], 'sig + assump ok'],
        'Orange': [colors[4], 'sig + assump not ok'],
        'Red': [colors[0],  'no sig + assump ok'],
        'Blue':[colors[1],  'no sig but non-param sig'],
        'Other': [colors[8], 'nothing sig']
    }
    
    legend_traces = []
    for label, color in legend_items.items():
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color[0]),
                legendgroup=label,
                showlegend=True,
                name=color[1]
            )
        )
    if project_loaded.meta == True:
        meta = 'M'
    else:
        meta = ''
    # Create the 2D plot
    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
                    layout=go.Layout(
                        title=dict(
                            text=f'Network of {meta}FG  :  Nb of features:{len(preprocessed_df)}  -  Nb of {meta}FG: {len(ion_df)}',
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        legend=dict(x=1, y=0.5, traceorder='normal'),
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    network_fig = fig
    return fig

    
# Definition of the layout to use
###############################################################################

def get_layout():
    return main_layout
    # project_loaded = cache.get('project_loaded')
    # if project_loaded != None:
    #     return main_layout
    # else:
    #     return dcc.Link('Go to home', href='/home')

    
layout = main_layout
