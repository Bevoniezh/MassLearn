# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:17:08 2024

@author: Ronan
"""

import os
import re
import sys
import dash
import glob
import json
import time
import uuid
import random
import scipy
import shutil
import threading
import logging
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from diskcache import Cache
import Modules.questions as Q
from collections import Counter
import Modules.convert as convert
from itertools import combinations
import Modules.cache_manager as cm
import Modules.grouping_tool as gt
import Modules.cleaning as cleaning
import Modules.features as features
import Modules.file_manager as fmanager
from pathlib import Path
from Modules import logging_config
import dash_bootstrap_components as dbc
from dash.dependencies import MATCH, ALL
from dash.exceptions import PreventUpdate
import networkx.algorithms.community as nx_comm
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, callback_context, callback

cache = Cache('./disk_cache')

dash.register_page(__name__)

logger = logging.getLogger(__name__)

SOFTWARE_WARNING_STYLE = {
    'maxWidth': '600px',
    'fontSize': '12px',
    'padding-left': '5px',
    'padding-right': '5px',
}

print(f'project cache up: {cache.get("project")}')

logo = cache.get('logo')

def get_layout():
    global line_count, current_project, sample_names, msn_df, project_name, erase_project
    global global_progress, start_time, estimated_total_time , failure, template_dict, tables_list, label_tables_list
    global treatment_groups, experiment_titles, skip, mzmine_process_start, template_to_proceed, exp_title_to_proceed
    global first_process_thread, second_process_thread, ms1_mzmine_instance, ms2_mzmine_instance, message
    global status, button_status, status2, message2, button_status2, feature_id
    
    line_count = 2 # the first number is directly displayed with the first layout container
    current_project = None
    sample_names = []
    msn_df = None
    project_name = None
    erase_project = None
    global_progress = 0                    
    start_time = None
    estimated_total_time = None
    failure = []
    template_dict = {}
    tables_list = []
    label_tables_list = []
    treatment_groups = {}
    experiment_titles = []
    skip = False 
    mzmine_process_start = False
    template_to_proceed = []
    exp_title_to_proceed = None
    first_process_thread = None
    second_process_thread = None
    ms1_mzmine_instance = None
    ms2_mzmine_instance = None
    message = None
    status = None
    button_status = None
    status2 = None
    message2 = None
    button_status2 = None
    feature_id = 0
    if cache.get('identity') is not None:
        logging_config.log_info(logger, 'Creation of a new project.')
        return main_layout
    else:
        return dcc.Link('Go to login', href='/login')

# Function to remove quotation marks surronding path if a user use function "copy as path" in Windows explorer
def normalize_path(input_path = ''):
    # Check if the path is enclosed in quotation marks and remove them
    if input_path == None:
        input_path = ''
    if input_path.startswith(("'", '"')) and input_path.endswith(("'", '"')):
        # Remove the first and last characters (the quotation marks)
        return input_path[1:-1]
    return input_path         

layout = dcc.Link('Go to login', href='/login')

line_count = 2 # the first number is directly displayed with the first layout container
current_project = None
sample_names = []
msn_df = None


class SoftwarePathError(RuntimeError):
    """Raised when an external software path is missing or invalid."""


def _get_configured_software_path(software_name: str, friendly_name: str) -> str:
    """Return a validated filesystem path for an external dependency.

    Parameters
    ----------
    software_name:
        Identifier used in :class:`Modules.cache_manager.Software_DashApp`.
    friendly_name:
        Human-readable label included in error messages.
    """

    soft = cm.Software_DashApp()
    raw_path = soft.get_path(software_name, default='').strip()

    if not raw_path:
        raise SoftwarePathError(
            f"{friendly_name} path is not configured. Please go back to the Login page and set it before continuing."
        )

    normalised = os.path.normpath(os.path.abspath(raw_path))

    if not os.path.exists(normalised):
        raise SoftwarePathError(
            f"The configured path for {friendly_name} does not exist: {normalised}. Update it from the Login page and try again."
        )

    return normalised


def _launch_external_software(executable_path: str) -> None:
    """Attempt to launch an external executable in a cross-platform manner."""

    if hasattr(os, "startfile"):
        os.startfile(executable_path)
        return

    try:
        subprocess.Popen([executable_path])
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise OSError(f"Unable to launch {executable_path}: {exc}") from exc


def _open_path_with_default_app(target_path: Path) -> None:
    """Open a file using the operating system's default application."""

    if hasattr(os, "startfile"):
        os.startfile(str(target_path))
        return

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target_path)])
        return

    subprocess.Popen(["xdg-open", str(target_path)])


def _session_log_path() -> Path:
    """Return the path to the runtime session log file."""

    module_root = Path(logging_config.__file__).resolve().parent
    return module_root.parent / "log.log"


# Definition on main layouts:
###############################################################################

title = dbc.Row(
    dbc.Col(html.H3('New project', style={'textAlign': 'center'}),
            className="d-flex justify-content-center align-items-center",
            style={'height': '50px'}),  # Adjust the height as needed
    className="w-100"
)

def create_separating_line(Separating_line_nb):
    separating_line = html.Div([
        html.Br(),
        html.Div([
            html.Hr(style={'display': 'inline-block', 'width': '45%'}),
            html.Span(Separating_line_nb, style={
                'display': 'inline-block',
                'position': 'relative',
                'top': '-1.7em',
                'padding': '0 0.5em',
                'font-weight': '200',  # Lighter font weight for thinner text
            }),
            html.Hr(style={'display': 'inline-block', 'width': '45%'}),
        ], style={'text-align': 'center'}),
        html.Br(),
    ])
    return separating_line


# Definition of navbar
###############################################################################

dropdownitems = [value for key, value in cache.get('project').items()]
for di in dropdownitems:
    di.disabled = True # disable all project in this pipeline

Project_menu = dbc.DropdownMenu(
            label="Create a new project",
            children=dropdownitems,
            id="up-project-menu", # up stands for untargeted pipeline
            nav=True,
        ),

navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.Img(src='/assets/logo.png', height="40px")),
                dbc.Col(
                    dbc.NavbarBrand("<- -        Untargeted Menu", className="ms-2",
                                    style={"fontSize": "16px"})  # smaller font
                ),
            ],
            align="center",
            className="g-0",
            ),
            href="/untargeted_menu",
            style={"textDecoration": "none"},
        ),
        dbc.Col(
            "Untargeted MS pipeline",
            width="auto",
            className="d-flex justify-content-center",
            style={"fontSize": "20px", "fontWeight": "bold"}  # bigger font
        ),
    ]),
    color="dark",
    dark=True,
    style={'height': '50px'},
)


# Callback to add new project to the list
@callback(
    [Output("up-project-menu", "children"),
    Output("up-project-menu", "label")],
    [Input("project-name-input", "valid"),
    Input('url', 'pathname')],
    [State("up-project-menu", "children"),
    State("project-name-input", "value"),
    State("up-project-menu", "label")],
    prevent_initial_call = True
)
def add_dropdown_item(valid, reload, existing_items, project_name, project_label):  
    if valid == None:
        dropdownitems = [value for key, value in cache.get('project').items()]
        for di in dropdownitems:
            di.disabled = True # disable all project in this pipeline
        label = 'Create a new project'
        return dropdownitems, label
    elif valid != None:
        project_dict = cache.get('project')
        new_item = dbc.DropdownMenuItem(project_name, id={"type": "project", "index": len(existing_items)}, n_clicks = 0)
        project_dict[f'p{len(project_dict)}'] = new_item
        cache.set('project', project_dict)
        if existing_items == []:
            existing_items = [new_item]
        else:
            existing_items.append(new_item)
        return existing_items, project_name

# function to trigger the callback for the newly created project
@callback(
    Output("output-container", "children"),
    [Input({"type": "project", "index": ALL}, "n_clicks")],
    prevent_initial_call=True
)
def item_clicked(args):
    ctx = dash.callback_context
    print('project')
    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return ""


# Definition of pop up Q&A
###############################################################################

popup_qa = html.Div([
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(id = 'untargeted-pipeline-modal_title')),
                dbc.ModalBody(id = 'untargeted-pipeline-modal_body'),
                dbc.ModalFooter(
                    html.Div([
                    dbc.Button(id="untargeted-pipeline-answer_a", color="secondary", n_clicks=0),
                    dbc.Button(id="untargeted-pipeline-answer_b", color="secondary", n_clicks=0),
                    dbc.Button(id="untargeted-pipeline-answer_c", color="secondary", n_clicks=0),
                                ], className="d-grid gap-2"), style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'alignItems': 'center',      # Center horizontally in the flex container
                                })
                    ], id="untargeted-pipeline-modal", is_open=False)
                ])

answer_save = pd.DataFrame()
answer_1 = ''
answer_2 = ''
answer_3 = ''
# TODO: delay automtically the closing of the modal.  (not is_open) close it
@callback(
    [Output("untargeted-pipeline-modal", "is_open"),
     Output("untargeted-pipeline-modal_title", "children"),
     Output("untargeted-pipeline-modal_body", "children"),
     Output("untargeted-pipeline-answer_a", "children"),
     Output("untargeted-pipeline-answer_b", "children"),
     Output("untargeted-pipeline-answer_c", "children"),
     Output("untargeted-pipeline-answer_a", "color"),
     Output("untargeted-pipeline-answer_b", "color"),
     Output("untargeted-pipeline-answer_c", "color"),
     Output("untargeted-pipeline-answer_a", "disabled"),
     Output("untargeted-pipeline-answer_b", "disabled"),
     Output("untargeted-pipeline-answer_c", "disabled"),
     Output("untargeted-pipeline-answer_a", "n_clicks"),
     Output("untargeted-pipeline-answer_b", "n_clicks"),
     Output("untargeted-pipeline-answer_c", "n_clicks")],
    [Input({"type": "popup", "index": ALL}, "children"),
     Input("untargeted-pipeline-answer_a", "n_clicks"),
     Input("untargeted-pipeline-answer_b", "n_clicks"),
     Input("untargeted-pipeline-answer_c", "n_clicks")],    
    [State("untargeted-pipeline-modal", "is_open"),
     State('untargeted-pipeline-switches-QA', 'value')],
    prevent_initial_call = True
)
def open_modal(popup, answer_a, answer_b, answer_c, is_open, switch_value):
    if popup[-2] == 'y' or is_open:
        if switch_value == True:
            cache.set('switch_QA_status', switch_value)
            global answer_save, answer_1, answer_2, answer_3
            q = Q.Questions()
            question_row = q.random_row
            if answer_save.empty:
                answer_save = question_row
                
                title = question_row['Theme'].iloc[0]
                question = question_row['Question'].iloc[0]
                correct = question_row['Correct Answer'].iloc[0] - 3 # we remove 3 because of the number of column in th question df
                answers = [question_row['Answer 1'].iloc[0], question_row['Answer 2'].iloc[0], question_row['Answer 3'].iloc[0]]
                correct_answer = answers[correct] 
                
                answer_1 = random.choice(answers)
                answer_1_color = "secondary"
                answers.remove(answer_1) # remove the chosen answer
                
                answer_2 = random.choice(answers)
                answer_2_color = "secondary"
                answers.remove(answer_2)
                
                answer_3 = answers[0]
                answer_3_color = "secondary"
            else:
                question_row = answer_save
                
                title = question_row['Theme'].iloc[0]
                question = question_row['Question'].iloc[0]
                correct = question_row['Correct Answer'].iloc[0] - 3 # we remove 3 because of the number of column in th question df
                answers = [question_row['Answer 1'].iloc[0], question_row['Answer 2'].iloc[0], question_row['Answer 3'].iloc[0]]
                correct_answer = answers[correct] 
                
                if answer_1 == correct_answer:
                    answer_1_color = "success"
                elif answer_1 != correct_answer and answer_a == 1: 
                    answer_1_color = "danger"
                else:
                    answer_1_color = "secondary"
                
                if answer_2 == correct_answer:
                    answer_2_color = "success"
                elif answer_2 != correct_answer and answer_b == 1: 
                    answer_2_color = "danger"
                else: 
                    answer_2_color = "secondary"
                
                if answer_3 == correct_answer:
                    answer_3_color = "success"
                elif answer_3 != correct_answer and answer_c == 1: 
                    answer_3_color = "danger"
                else: 
                    answer_3_color = "secondary"
            
            if answer_a or answer_b or answer_c:
                answer_save = pd.DataFrame()
                return True, title, question, answer_1, answer_2, answer_3, answer_1_color, answer_2_color, answer_3_color, True, True, True, 0, 0, 0
                
            else:      
                return (not is_open), title, question, answer_1, answer_2, answer_3, "secondary", "secondary", "secondary", None, None, None, 0, 0, 0 # return invert of the current state of the modal, and the modal parameters
        
    raise PreventUpdate()


            
# 1- Input for directory path
###############################################################################            
@callback(
    [Output('project-name', 'children'),
    Output("dir-output", "children"),
    Output("dir-input", "valid"),
    Output("dir-input", "invalid"),
    Output("dir-input", "disabled"),
    Output({'type': 'popup', 'index': 0}, 'children')
    ],
    [Input("dir-input", "n_submit")],
    [State("dir-input", "value")],
    prevent_initial_call = True
)
def validate_dir(n_submit, path_to_check):
    global line_count, existing_popups
    if n_submit and path_to_check!= None:
        path_to_check = normalize_path(path_to_check)
        # Add your validation logic here
        if os.path.exists(path_to_check) and os.path.isdir(path_to_check):
            cache.set('project_path', path_to_check)
            logging_config.log_info(logger, '%s added for new project', path_to_check)
            new_popup = html.Div(children = '', id={"type": "popup", "index": 1}, style={'display': 'none'})
            separating_line = create_separating_line(line_count)
            line_count += 1
            return [separating_line, project_part, new_popup], "Path successfully loaded!", True, None, True, 'y'
        else:
            return "", "Not a valid path, make sure the folder exist, or verify if the path is not a file", None, True, False, 'n'
    raise PreventUpdate()

first_line = html.Div([
    html.Br(),
    html.Div([
        html.Span(1, style={
            'display': 'inline-block',
            'position': 'relative',
            'top': '-1.7em',
            'padding': '0 0.5em',
            'font-weight': '200',  # Lighter font weight for thinner text
        }),
    ], style={'text-align': 'center'}),
    html.Br(),
])

dir_input = html.Div([
    first_line,
    html.Div(id = {"type": "popup", "index": 0}),
    html.Div([
        html.H5('Indicate the EMPTY folder where you want to save your project, by giving the path, and press Enter', style={'textAlign': 'center'}),
        dbc.Input(id="dir-input", valid = None, placeholder=r"C:\Users\Arthur\LCMS_project1", type="text", style={'maxWidth': '600px'}),
        html.Br(),
        html.P(id="dir-output"),
        
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',      # Center horizontally in the flex container
        }),
        html.Div(id = 'project-name')
        ])
  

# 2- Input for project name
###############################################################################
project_name = None
erase_project = None
@callback(
    [Output('raw-dir', 'children'),
     Output("project-name-output", "children"),
     Output("project-name-input", "valid"),
     Output("project-name-input", "invalid"),
     Output("project-name-input", "disabled"),
     Output({'type': 'popup', 'index': 1}, 'children')
     ],
    [Input("project-name-input", "n_submit")],
    [State("project-name-input", "value")],
    prevent_initial_call = True
)                  
def validate_project(n_submit, input_project_name):
    global current_project, project_name, erase_project, line_count
    if n_submit:
        if input_project_name:
            if len(input_project_name) > 4:
                pathfolder = cache.get('project_path')
                valid = fmanager.FileManager(input_project_name, pathfolder, 'data', 'txt') # 'data' and 'txt' are just mock information because we wont save those information yet
                test = valid.is_valid_filename(input_project_name) # check for special character presence
                if test == True:
                    logging_config.log_info(logger, 'New project name: %s', input_project_name)
                    project = fmanager.Project(input_project_name, pathfolder, Newproject = True) # create new project
                    project_file = fmanager.FileManager(input_project_name, pathfolder, project, 'masslearn')
                    subfolder_mzml = os.path.join(pathfolder, 'mzML') # verify if there is a subfolder for denoised file
                    if not os.path.exists(subfolder_mzml):
                        os.makedirs(subfolder_mzml)
                    project.mzml_folder_path = subfolder_mzml 
                    
                    subfolder_features = os.path.join(pathfolder, 'feature') # verify if there is a subfodler for denoised file
                    if not os.path.exists(subfolder_features):
                        os.makedirs(subfolder_features)
                        os.makedirs(subfolder_features + '/noise')
                    else:
                        if not os.path.exists(subfolder_features + '/noise'):
                            os.makedirs(subfolder_features + '/noise')
    
                    project.featurepath = subfolder_features  
                    
                    subfolder_plot = os.path.join(pathfolder, 'plot')
                    if not os.path.exists(subfolder_plot):
                        os.makedirs(subfolder_plot)
                    project.plotpath = subfolder_plot 
                    if project_file.handle_existing_file_dash():
                        project_file.saving_file_dash(project=project)
                        current_project = project
                        cache.set('project_loaded', current_project)
                        separating_line = create_separating_line(line_count)
                        line_count += 1
                        new_popup = html.Div(children = '', id={"type": "popup", "index": 2}, style={'display': 'none'})
                        return [separating_line, new_popup, raw_dir], f'{input_project_name}.masslearn created!', True, None, True, 'y'
                    else:
                        if erase_project == None:
                            erase_project = True
                            return "", "This project name already exists. Press Enter again to erase it and create a new one." , None, True, False, 'n'
                        elif erase_project:
                            project_file.saving_file_dash(project=project)
                            current_project = project
                            cache.set('project_loaded', current_project)
                            separating_line = create_separating_line(line_count)
                            line_count += 1
                            new_popup = html.Div(children = '', id={"type": "popup", "index": 2}, style={'display': 'none'})
                            return [separating_line, new_popup, raw_dir], f'New {input_project_name}.masslearn created!', True, None, True, 'y'
                    
                else:
                    "", r"No special characters like .?\:; etc , only numbers, letters and _" , None, True, False, 'n'
            else:
                return "", "Your project name needs at least 5 letters", None, True, False, 'n'
    else:
        raise PreventUpdate()

project_part = html.Div([
                html.Div([
                    html.H5('Give a name to your project, and press Enter', style={'textAlign': 'center'}),
                    dbc.Input(id="project-name-input", valid = None, placeholder= "Project_1", type="text", style={'maxWidth': '600px'}),
                    html.Br(),
                    html.P(id="project-name-output"),
                    html.Br(),
                    ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',      # Center horizontally in the flex container
                        }),
                html.Div(id = 'raw-dir'),                
                    ])    

                    
# 3- Select raw dir or continue with mzml files
###############################################################################
RAW_FILE_TYPES = {
    'waters': {
        'label': 'Waters (.raw folders)',
        'placeholder': r"C:\\Users\\Arthur\\Waters_raw",
        'hint': 'Waters .raw folders containing _FUNC001.DAT files',
    },
    'thermo': {
        'label': 'Thermo (.raw files/folders)',
        'placeholder': r"C:\\Users\\Arthur\\Thermo_raw",
        'hint': 'Thermo .raw files or folders',
    },
    'bruker': {
        'label': 'Bruker (.d folders)',
        'placeholder': r"C:\\Users\\Arthur\\Bruker_d",
        'hint': 'Bruker .d folders',
    },
    'sciex': {
        'label': 'SCIEX (.wiff files)',
        'placeholder': r"C:\\Users\\Arthur\\Sciex_wiff",
        'hint': 'SCIEX .wiff files (with associated .scan files if applicable)',
    },
}
DEFAULT_RAW_FILE_TYPE = 'waters'

@callback(
    Output('raw-dir-input', 'placeholder'),
    Output('raw-dir-guidance', 'children'),
    Input('raw-file-type', 'value')
)
def update_raw_dir_guidance(file_type):
    file_type = (file_type or DEFAULT_RAW_FILE_TYPE).lower()
    info = RAW_FILE_TYPES.get(file_type, RAW_FILE_TYPES[DEFAULT_RAW_FILE_TYPE])
    guidance = f"Expected: {info['hint']}." if info.get('hint') else ''
    return info.get('placeholder', r"C:\\Users\\Arthur\\Raw-files"), guidance

# Function to get all vendor files from a directory
def get_raw_files(Directory, file_type = DEFAULT_RAW_FILE_TYPE):
    rawfiles = []
    file_type = (file_type or DEFAULT_RAW_FILE_TYPE).lower()
    for root, dirs, files in os.walk(Directory):
        if file_type in ['waters', 'thermo']:
            for d in dirs:
                if d.lower().endswith('.raw'):
                    rawfiles.append(os.path.join(root, d))
            if file_type == 'thermo':
                for f in files:
                    if f.lower().endswith('.raw'):
                        rawfiles.append(os.path.join(root, f))
        elif file_type == 'bruker':
            for d in dirs:
                if d.lower().endswith('.d'):
                    rawfiles.append(os.path.join(root, d))
        elif file_type == 'sciex':
            for f in files:
                if f.lower().endswith('.wiff') or f.lower().endswith('.wiff2'):
                    rawfiles.append(os.path.join(root, f))
    return rawfiles

# Function to check if there are vendor files in the folder
def check_raw_contents(dir_path, file_type = DEFAULT_RAW_FILE_TYPE):
    file_type = (file_type or DEFAULT_RAW_FILE_TYPE).lower()
    try:
        entries = os.listdir(dir_path)
    except FileNotFoundError:
        return False
    if file_type == 'waters':
        for entry in entries:
            if entry.lower().endswith('.raw'):
                full_path = os.path.join(dir_path, entry)
                if os.path.isdir(full_path):
                    for subentry in os.listdir(full_path):
                        if subentry == '_FUNC001.DAT':
                            return True
        return False
    elif file_type == 'thermo':
        return any(entry.lower().endswith('.raw') for entry in entries)
    elif file_type == 'bruker':
        return any(entry.lower().endswith('.d') for entry in entries)
    elif file_type == 'sciex':
        return any(entry.lower().endswith('.wiff') or entry.lower().endswith('.wiff2') for entry in entries)
    return False

@callback(
    [Output("mzml-progress", "value"),
     Output("mzml-progress", "style"),
     Output("mzml-alternative", "children"),
     Output("mzml-alternative", "disabled"),
     Output('mzml-interval-component', 'disabled')],
    [Input("mzml-alternative", "n_clicks"),
     Input('mzml-interval-component', 'n_intervals'),
     State('convert-raw', 'children')],
     prevent_initial_call = True
    )
def loading_buttom_mzml(n_clicks, n, state):
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "mzml-alternative":
        return 0, {"height": "3px", "width": "300px"}, [dbc.Spinner(size="sm"), " Loading the files..."], True, None
    elif button_id == "mzml-interval-component":
        if state == '':
            global mzml_loading
            if mzml_loading == -1:
                return 0, {'display':'none'}, ["Error! Try again."], None, True
            else:
                return mzml_loading, dash.no_update, dash.no_update, True, None
        else:
            return 100, dash.no_update, ["mzML files loaded."], True, True
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True
    
    
mzml_alternative = None
mzml_loading = 0
@callback(
    [Output('convert-raw', 'children'),
     Output("raw-dir-output", "children"),
     Output("raw-dir-input", "valid"),
     Output("raw-dir-input", "invalid"),
     Output("raw-dir-input", "disabled"),
     Output({'type': 'popup', 'index': 2}, 'children')
     ],
    [Input("raw-dir-input", "n_submit"),
     Input("mzml-alternative", "n_clicks")],
    [State("raw-dir-input", "value"),
     State('raw-file-type', 'value')],
    prevent_initial_call = True
)
def validate_raw_input(n_submit, n_clicks, path_to_check, file_type):
    global current_project, mzml_alternative, line_count, mzml_loading
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not ctx.triggered:
        raise PreventUpdate

    if button_id == 'mzml-alternative':
        current_project.mzml_files_path = sorted(
            filename
            for filename in glob.glob(os.path.join(current_project.mzml_folder_path, '*'))
            if filename.lower().endswith('.mzml')
        )
        if len(current_project.mzml_files_path) == 0:
            mzml_loading = -1 # tag for error of files
            return dash.no_update, "No files detected in the /mzML folder! Please copy your files before pushing the button.", None, None, True, 'y'
        elif (len(current_project.mzml_files_path)<3):
            mzml_loading = 100
            return dash.no_update, "MassLearn detect less than 3 files. You need at least one blank, one sample for treatment 1 and one sample for treamtment 2 to have any an analysis.", None, None, True, 'y'

        mzml_alternative = True
        current_project.sample_names = []
        current_project.files_spectra = {}
        current_project.raw_files_path = []
        cache.set('project_loaded', current_project)
        mzml_loading = 100
        separating_line = create_separating_line(line_count)
        line_count += 1
        new_popup = html.Div(children = '', id={"type": "popup", "index": 4}, style={'display': 'none'})
        return [separating_line, new_popup, noise_threshold], f"{len(current_project.mzml_files_path)} files detected.", None, None, True, 'y'

    if n_submit:
        path_to_check = normalize_path(path_to_check)
        file_type = (file_type or DEFAULT_RAW_FILE_TYPE).lower()
        info = RAW_FILE_TYPES.get(file_type, RAW_FILE_TYPES[DEFAULT_RAW_FILE_TYPE])
        # Add your validation logic here
        if os.path.exists(path_to_check) and os.path.isdir(path_to_check) and check_raw_contents(path_to_check, file_type):
            logging_config.log_info(logger, 'Raw files path: %s added.', path_to_check)

            current_project.raw_folder_path = path_to_check
            current_project.raw_file_type = file_type
            current_project.raw_files_path = get_raw_files(path_to_check, file_type)
            mzml_alternative = False
            cache.set('project_loaded', current_project)
            separating_line = create_separating_line(line_count)
            line_count += 1
            new_popup = html.Div(children = '', id={"type": "popup", "index": 3}, style={'display': 'none'})
            return [separating_line, new_popup, convert_raw], "Path successfully loaded!", True, None, True, 'y'
        else:
            return "", f"Not a valid path! Make sure the folder exists, verify it is not a file, and confirm it contains at least one {info['hint']}.", None, True, False, 'n'

    raise PreventUpdate()

raw_dir = html.Div([
            html.Div([
                html.H5('Are your files in vendor format or in .mzML format?', style={'textAlign': 'center'}),
                html.Br(),
                html.H6('(1) Select your vendor format, point to the folder containing the raw data, and press Enter', style={'textAlign': 'center'}),
                dbc.ListGroupItem('WARNING! For Waters data, the conversion procedure removes "_FUNC003" related files (.raw) corresponding to lockspray references. We recommend converting a copy of your raw data rather than the originals.', color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px',}),
                html.Br(),
                dbc.RadioItems(
                    id='raw-file-type',
                    options=[{'label': value['label'], 'value': key} for key, value in RAW_FILE_TYPES.items()],
                    value=DEFAULT_RAW_FILE_TYPE,
                    inputClassName='me-2',
                    labelClassName='d-block',
                    className='mb-2',
                    style={'maxWidth': '600px'}
                ),
                html.Div(id='raw-dir-guidance', className='text-muted', style={'fontSize': '12px', 'maxWidth': '600px', 'textAlign': 'center'}),
                html.Br(),
                dbc.Input(id="raw-dir-input", valid = None, placeholder=r"C:\Users\Arthur\Raw-files", type="text", style={'maxWidth': '600px'}),
                html.Br(),
                dcc.Interval(id='mzml-interval-component', interval=500, n_intervals=0, max_intervals=-1, disabled = True),  # Checks every second
                html.Br(),
                html.H5('... or ...', style={'textAlign': 'center'}),
                html.Br(),
                html.H6('(2) Your files are already in .MZML format .', style={'textAlign': 'center'}),
                html.H6('Copy them to /mzML folder which is present in your project folder. Then click on "Verify my .mzML files" button.', style={'textAlign': 'center'}),
                html.Br(),
                dbc.Button('Verify my .mzML files', id = "mzml-alternative", color="primary", n_clicks=0, style={"width": "300px"}),
                dbc.Progress(id="mzml-progress", color="success", style={'display':'none'}, className='mt-1'),
                html.Br(),
                dbc.Tooltip(
                    "MassLearn verifies that the selected folder contains at least one file matching the chosen vendor format.",
                    target="raw-dir-input",  # ID of the component to which the tooltip is attached
                    placement="left"),
                html.P(children = '\n', id="raw-dir-output"),
                html.Br(),
                ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',      # Center horizontally in the flex container
                    }),
            html.Div(id = 'convert-raw', children = '')
                ])
                    
                    
# 4- Convert raw files in mzML          
###############################################################################
@callback(
    [Output('noise-part', 'children'),
     Output("convert-range-button", "disabled"),
     Output("convert-input", "disabled"),
     Output({'type': 'popup', 'index': 3}, 'children')],
    [Input("convert-range-button", "n_clicks")],
    [Input("convert-input", "value")],
    prevent_initial_call = True
)
def convert_raw_input(n_clicks, value):
    global current_project, line_count
    if n_clicks:
        cache.set('raw_range', value)
        current_project.rt_range = value
        cache.set('project_loaded', current_project)

        logging_config.log_info(logger, 'Min rt: %s and max rt: %s.', value[0], value[1])
        separating_line = create_separating_line(line_count)
        line_count += 1
        new_popup = html.Div(children = '', id={"type": "popup", "index": 4}, style={'display': 'none'})
        return [separating_line, new_popup, noise_threshold], True, True, 'y'
    raise PreventUpdate()
       
                          
@callback(
    Output("software-msg", "children"),
    [Input("open-seems-btn", "n_clicks")]
)
def open_seems(n_clicks):
    if n_clicks > 0:
        try:
            seems_path = _get_configured_software_path('SeeMS', 'SeeMS (seems.exe)')
        except SoftwarePathError as exc:
            logging_config.log_warning(logger, str(exc))
            return dbc.ListGroupItem(str(exc), color="warning", style=SOFTWARE_WARNING_STYLE)

        try:
            _launch_external_software(seems_path)
            return "SeeMS opened"
        except FileNotFoundError as exc:
            logging_config.log_error(logger, 'SeeMS executable not found: %s', exc)
            return dbc.ListGroupItem(
                "SeeMS executable cannot be found at the configured location. Please verify the path from the Login page.",
                color="danger",
                style=SOFTWARE_WARNING_STYLE,
            )
        except OSError as exc:  # pragma: no cover - defensive
            logging_config.log_error(logger, 'Failed to open SeeMS: %s', exc, exc_info=True)
            return "Failed to open SeeMS. Try to open it externally."

    raise PreventUpdate()

                                
convert_raw = html.Div([ 
                html.Div([        
                    dbc.ListGroupItem("""The first step of the pipeline is to convert vendor-specific raw files (Waters, Thermo, Bruker, SCIEX) into the open mzML format. To perform this, we also define a new elution time range, default is from 50 sec to 360 sec.\
                \n\n # TIP # Usually in LCMS, we use a gradient of compounds in the mobile phase. For example in reverse phase, different \
                proportion of Acetonitrile and Methanol from the begining to the end of the elution. This change the compound affinity in selectivity, retention time, peak shape,signal intensity and solubility. \
                As a consequence, polar compounds reverse phase tends to elute all together at the beggining and more apolar ones together at the end. \
                That means there are generally two cluster of compounds signals, difficult to separate, at the beggining and the end of the elution we do not want to analyze. They for sure\
                increase the comptational burden for nothing. Reason why it is better to remove those parts, before 50 sec and after 360 sec depending on your LC design.\
                """, color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px',}),
                    html.Br(),
                    
                    dbc.Alert([
                        html.Div([
                            html.H6("Warning", style={'margin-top': '0px', 'margin-bottom': '5px'}),  # Reduce bottom margin
                            html.P("Open SeeMS, load the raw vendor files and estimate min and max retention time.", style={'margin-top': '0px', 'margin-bottom': '0px'}),
                            # Reduced top and bottom margins
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',
                        }),
                        html.Hr(style={'margin-top': '10px', 'margin-bottom': '10px'}),  # Reduce margin around <hr>
                        html.Div([
                            html.Button(
                                children=[html.Img(src='/assets/seems.png', style={'width': 'auto', 'height': '60px'})],
                                id='open-seems-btn',
                                style={'background': 'none', 'border': 'none', 'padding': '0', 'margin-top': '0px', 'margin-bottom': '0px'},
                                # Reduced padding, added reduced top and bottom margins
                                n_clicks=0),
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',
                        }),
                    ], color="warning", style={'padding': '0.5rem 1rem', 'width' : '600px'}), 
                    
                    html.Div(id="software-msg"),
                    html.H1(''),
                    html.H5('Indicate your conversion range and click "Convert"', style={'textAlign': 'center'}),
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',      # Center horizontally in the flex container
                    }),
                    html.Div([  
                    dcc.RangeSlider(
                            id='convert-input',
                            min=0,  # Minimum value of the range slider
                            max=1000,  # Maximum value of the range slider
                            marks={i: {'label': '{} s'.format(i), 'style': {'color': 'white'}} for i in range(0, 1001, 100)},
                            step=1,  # Step size
                            value=[50, 360],  # Default selected range
                            allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": True, "style": {"color": "LightSteelBlue", "fontSize": "20px"}, "transform": "convertSecondsToMinSec", "template": "{value}"},
                        )], style = {'width' : '600px', 'margin-left':'auto', 'margin-right':'auto'}),
                    html.Div([ 
                    html.Br(),
                    dbc.Button('Save conversion', id = "convert-range-button", color="primary", n_clicks=0),       
                    dbc.Tooltip(
                        "Minimum rt and maximum rt (sec) to crop your raw files based on the selected vendor format",
                        target="convert-input",  # ID of the component to which the tooltip is attached
                        placement="left"),
                    
                    html.Br(),
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',      # Center horizontally in the flex container
                    }),
                    html.Div(id = 'noise-part'),
                ])
                          
# 5- Define Noise threshold      
###############################################################################
@callback(
    [Output('ms-noise', 'children'),
     Output('noise-trace-button', 'disabled'),
     Output({'type': 'popup', 'index': 4}, 'children')
     ],
    [Input("noise-trace-button", "n_clicks")],
    [State("noise-threshold", "value")],
    prevent_initial_call = True
)                        
def validate_noise_raw_input(n_clicks, threshold):
    global current_project, line_count
    if n_clicks:
        if 0 < threshold < 101:
            current_project.noise_trace_threshold = threshold
            cache.set('project_loaded', current_project)

            logging_config.log_info(logger, 'Noise trace threshold: %s', threshold)
            separating_line = create_separating_line(line_count)
            line_count += 1
            new_popup = html.Div(children = '', id={"type": "popup", "index": 5}, style={'display': 'none'})
            return [separating_line, new_popup, ms_noise], True, 'y'
        else:
            raise PreventUpdate()
    raise PreventUpdate()

noise_threshold = html.Div([ 
                    html.Div([        
                        dbc.ListGroupItem("""We need to remove noise traces. To remove it, we basically remove the masses which appear to be present \
            along a minimum percentage of the elution time. E.g, if a mass is present in at least 20% \
            of the scans, it is highly possible it is a noise trace. In normal situation, it will be lower. By default, when the total \
            elution time kept in your study design is 5-6 min from the beggining to the end, we consider 20% as a satisfying threshold to detect noise. \
            If your elution time range is thinner, you should increase the percentage because \
            the elution time taken by a significant feature is relatively higher compared to what \
            we found in a wider elution time range, vice versa. \
            Also, all mass with a > x.8 decimal value are deleted because expected to be non-natural\
            \n!!! Be careful, the lower your noise trace detection threshold, the longer it will be to process it !!!\
                    """, color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px',}),
                        html.Br(),
                        html.H1(''),
                        html.H5('Enter the Noise trace threshold and click on "Confirm noise trace"', style={'textAlign': 'center'}),                        
                        dbc.InputGroup([ 
                            dbc.InputGroupText("Noise trace threshold (%)"),  # Adding a label
                            dbc.Input(id='noise-threshold', 
                                      type="number", 
                                      value=20, 
                                      min=1,    # Minimum value allowed
                                      max=100,  # Maximum value allowed
                                      step=1,   # Increment step
                                      maxLength=3,  # Limit input to 3 characters
                                      size="sm"),  # Set the size of the input
                            dbc.InputGroupText("%"),  # Adding "%" at the end of the input bar
                                ], style={'maxWidth': '450px'}),
                        html.Br(),
                        dbc.Button("Confirm noise trace", id = "noise-trace-button", color="primary", n_clicks=0),
                        dbc.Tooltip(
                            "If a m/z (+- 0.005 Da) is detected > Threshold % of the whole elution time range, this m/z (+- 0.005 Da) is deleted).",
                            target="noise-threshold",  # ID of the component to which the tooltip is attached
                            placement="right"),
                        html.Br(),
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',      # Center horizontally in the flex container
                        }),
                    html.Div(id = 'ms-noise')
                    ])
                        
# 6- Define MS1 and MS2 basic noise cut off    
###############################################################################
@callback(
    [Output('conversion-progress-part', 'children'),
     Output('ms-noise-button', 'disabled'),
     Output({'type': 'popup', 'index': 5}, 'children')
     ],
    [Input("ms-noise-button", "n_clicks")],
    [State("ms1-noise", "value"),
     State("ms2-noise", "value")],
    prevent_initial_call = True
)
def validate_ms_noise_input(n_clicks, ms1, ms2):
    global current_project, line_count, mzml_alternative, global_progress, failure, start_time, estimated_total_time
    if n_clicks:
        if 0 <= ms1 and 0 <= ms2:
            if not mzml_alternative:
                try:
                    proteowizard_path = _get_configured_software_path('ProteoWizard', 'ProteoWizard (msconvert.exe)')
                except SoftwarePathError as exc:
                    logging_config.log_warning(logger, str(exc))
                    warning = dbc.ListGroupItem(str(exc), color="warning", style=SOFTWARE_WARNING_STYLE)
                    return warning, False, 'n'
            else:
                proteowizard_path = None

            current_project.ms1_noise = ms1
            current_project.ms2_noise = ms2
            cache.set('project_loaded', current_project)
            logging_config.log_info(
                logger,
                'ms1 noise threshold: %s and ms2 noise threshold: %s',
                ms1,
                ms2,
            )

            global_progress = 0
            failure = []
            start_time = None
            estimated_total_time = None

            if mzml_alternative:
                processing_thread = threading.Thread(target=process_mzml_files, args=(current_project.mzml_files_path,))
            else:
                processing_thread = threading.Thread(target=process_files, args=(current_project.raw_files_path, proteowizard_path))

            processing_thread.start()
            separating_line = create_separating_line(line_count)
            line_count += 1
            new_popup = html.Div(children = '', id={"type": "popup", "index": 6}, style={'display': 'none'})
            return [separating_line, new_popup, progress], True, 'y'
        else:
            raise PreventUpdate()
    raise PreventUpdate()
ms_noise = html.Div([ 
                html.Div([        
                    dbc.ListGroupItem("""For convenience, all masses with an intensity (or count) lower than 400 for MS1 and 200 for MS2 will be removed by default. \
        Thoses masses represent the background noise, which is different from the noises traces. \
        We HIGHLY recommend to remove those low m/z counts, otherwise further steps will take a lot of time.\
                """, color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px',}),
                    html.Br(),
                    html.H1(''),
                    html.H5('Enter the Noise threshold and click on "Begin processing"', style={'textAlign': 'center'}),
                    dbc.InputGroup([ 
                        dbc.InputGroupText("MS level 1 minimum intensity detectable"),  # Adding a label
                        dbc.Input(id='ms1-noise', 
                                  type="number", 
                                  value=400, 
                                  min=0,    # Minimum value allowed
                                  step=1,   # Increment step
                                  size="sm"),  # Set the size of the input
                        dbc.InputGroupText("ions"),  # Adding "%" at the end of the input bar
                            ], style={'maxWidth': '690px'}),
                    html.Br(),
                    dbc.InputGroup([ 
                        dbc.InputGroupText("MS level 2 minimum intensity detectable"),  # Adding a label
                        dbc.Input(id='ms2-noise', 
                                  type="number", 
                                  value=200, 
                                  min=0,    # Minimum value allowed
                                  step=1,   # Increment step
                                  size="sm"),  # Set the size of the input
                        dbc.InputGroupText("ions"),  # Adding "%" at the end of the input bar
                            ], style={'maxWidth': '690px'}),
                    html.Br(),
                    dbc.Button("Begin processing", id = "ms-noise-button", color="primary", n_clicks=0),
                    dbc.Tooltip(
                        "Below this limit of ion count, all signal from MS level 1 (mostly potential precursor ions) detected by the Mass Spectrometer will be removed natively from the files.",
                        target='ms1-noise',  # ID of the component to which the tooltip is attached
                        placement="right"),
                    dbc.Tooltip(
                        "Below this limit of ion count, all signal from MS level 2 (mostly fragment ions) detected by the Mass Spectrometer will be removed natively from the files.",
                        target='ms1-noise',  # ID of the component to which the tooltip is attached
                        placement="right"),
                    html.Br(),
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',      # Center horizontally in the flex container
                    }),
                    
                    html.Div(id = 'conversion-progress-part')
                ])
                        
# 7- Conversion, denoising and progress bar
###############################################################################
# Global variable to track progress
global_progress = 0                    
start_time = None
estimated_total_time = None
failure = []
def save_project():
    global current_project
    saving = current_project.save()

def process_files(files, proteowizard_path):
    global global_progress
    global current_project
    global start_time
    global estimated_total_time
    global sample_names
    global failure

    total_files = len(files)

    if total_files == 0:
        global_progress = 100
        estimated_total_time = 0
        failure.append('No files provided for conversion.')
        return

    start_time = time.time()
    file_type = getattr(current_project, 'raw_file_type', DEFAULT_RAW_FILE_TYPE)
    current_project.sample_names = []
    current_project.mzml_files_path = []
    current_project.files_spectra = {}
    for nb, file in enumerate(files):
        try:
            # Define all types of file name, path, etc
            rawfile_path = file
            rawfile_path_noext, _ = os.path.splitext(file)
            sample_name = os.path.basename(rawfile_path_noext)
            rawfolder_mzmlfile_path = rawfile_path_noext + '.mzML'
            mzmlfile_basename = os.path.basename(rawfolder_mzmlfile_path)
            mzmlfolder_mzmlfile_path = os.path.join(current_project.mzml_folder_path, mzmlfile_basename)
            current_project.mzml_files_path.append(mzmlfolder_mzmlfile_path)
            current_project.sample_names.append(sample_name)
            # Convert the file
            raw_to_convert = convert.RawToMZml([rawfile_path], current_project.rt_range[0], current_project.rt_range[1], file_type)
            raw_to_convert.convert_file(proteowizard_path)
            size_in_bytes = os.path.getsize(rawfolder_mzmlfile_path)
            size_in_kb = size_in_bytes / 1024 # Convert the size from bytes to kilobytes (1 KB = 1024 bytes)
            if size_in_kb < 300: # below 300kB it is not a satisfying mzml file
                logging_config.log_warning(logger, '%s too small, removed from the file list.', sample_name)
            else:
                # Denoise the file
                spectra = cleaning.Spectra(rawfolder_mzmlfile_path) # take all the spectra data
                spectra.extract_peaks(current_project.ms1_noise, current_project.ms2_noise) # peak variable is only here to allow loading bar to not be disturbe
                to_denoise_file = cleaning.Denoise(rawfolder_mzmlfile_path, current_project.featurepath)
                denoised_spectra, ms1_spectra, ms2_spectra = to_denoise_file.filtering(
                    spectra,
                    current_project.noise_trace_threshold,
                    Dash_app=True,
                ) # denoised_spectra is the spectra class object from cleaning module, containing lots of attributes. Encoded are just variables with basice array denoised
                current_project.files_spectra[sample_name] = ms1_spectra, ms2_spectra ##### Important: here you find the spectra files, use it to plot the features
                # Move and delete the files
                if os.path.exists(mzmlfolder_mzmlfile_path):
                    os.remove(mzmlfolder_mzmlfile_path)
                    shutil.move(rawfolder_mzmlfile_path, current_project.mzml_folder_path)
                else:
                    shutil.move(rawfolder_mzmlfile_path, current_project.mzml_folder_path)
                if os.path.exists(rawfolder_mzmlfile_path):
                    os.remove(rawfolder_mzmlfile_path)

                logging_config.log_info(logger, '%s converted and denoised.', sample_name)
        except Exception as exc:
            rawfile_path = file
            rawfile_path_noext, _ = os.path.splitext(file)
            sample_name = os.path.basename(rawfile_path_noext)
            logging_config.log_exception(
                logger,
                '%s conversion failure.',
                sample_name,
                exception=exc,
            )
            print(f'{sample_name} conversion failure.')

            failure.append(sample_name)

        global_progress = int(((nb + 1) / (total_files)) * 100)
        if global_progress == 0:
            global_progress == 1
        elapsed_time = time.time() - start_time
        try:
            estimated_total_time = elapsed_time / (global_progress / 100)
        except Exception:
            estimated_total_time = elapsed_time / 0.1
    sample_names = current_project.sample_names


def process_mzml_files(files):
    global global_progress
    global current_project
    global start_time
    global estimated_total_time
    global sample_names
    global failure

    total_files = len(files)

    if total_files == 0:
        global_progress = 100
        estimated_total_time = 0
        failure.append('No mzML files detected for denoising.')
        return

    start_time = time.time()
    current_project.sample_names = []
    current_project.files_spectra = {}

    for nb, file in enumerate(files):
        rawfile_path_noext, _ = os.path.splitext(file)
        sample_name = os.path.basename(rawfile_path_noext)
        try:
            spectra = cleaning.Spectra(file)
            spectra.extract_peaks(current_project.ms1_noise, current_project.ms2_noise)
            to_denoise_file = cleaning.Denoise(file, current_project.featurepath)
            denoised_spectra, ms1_spectra, ms2_spectra = to_denoise_file.filtering(
                spectra,
                current_project.noise_trace_threshold,
                Dash_app=True,
            )
            current_project.sample_names.append(sample_name)
            current_project.files_spectra[sample_name] = ms1_spectra, ms2_spectra
            logging_config.log_info(logger, '%s denoised.', sample_name)
        except Exception:
            logging_config.log_error(logger, '%s denoising failure.', sample_name)
            print(f'{sample_name} denoising failure.')
            failure.append(sample_name)

        global_progress = int(((nb + 1) / (total_files)) * 100)
        if global_progress == 0:
            global_progress == 1
        elapsed_time = time.time() - start_time
        try:
            estimated_total_time = elapsed_time / (global_progress / 100)
        except Exception:
            estimated_total_time = elapsed_time / 0.1

    sample_names = current_project.sample_names

@callback(
    [Output("template-part", "children"),
     Output("denoise-progress", "value"), 
     Output("denoise-progress", "label"), 
     Output("denoise-progress", "animated"),
     Output("conversion-info", "children"),
     Output('denoise-progress-interval', 'disabled')],
    Input("denoise-progress-interval", "n_intervals")
)
def update_conversion_progress(n):
    # Use the global progress
    global global_progress, line_count
    global current_project
    global start_time
    global estimated_total_time
    global failure
    global mzml_alternative
    
    progress = global_progress
    
    if start_time != None:
        elapsed_time = time.time() - start_time
    else:
        elapsed_time = 0
    if progress > 0:
        time_remaining = estimated_total_time - elapsed_time
        minutes_remaining = int(time_remaining // 60)
        seconds_remaining = int(time_remaining % 60)
        if seconds_remaining < 5:
            seconds_remaining = 5
        title_status = f"Estimated time remaining: {minutes_remaining} min {seconds_remaining} sec"
    else:
        title_status = "Calculating time remaining..."
    
    if progress < 100:
        return "", progress, f"{progress}%" if progress > 0 else "", None, title_status, False
    else:
        if failure == []:
            cache.set('project_loaded', current_project)
            thread_project = threading.Thread(target=save_project)
            thread_project.start()
            separating_line = create_separating_line(line_count)
            line_count += 1
            if mzml_alternative:
                completion_message = "Denoising complete"
            else:
                completion_message = "Conversion and denoising complete"
            return [separating_line, template_part], progress, f"{progress}%" if progress > 0 else "", None, completion_message, True
        else:
            separating_line = create_separating_line(line_count)
            line_count += 1
            list_fail_samples = ', '.join(failure)
            if mzml_alternative:
                error_info = f'Error happened while denoising: {list_fail_samples}.'
                completion_message = "Denoising complete with errors."
            else:
                error_info = f'Error happened while processing: {list_fail_samples}.'
                completion_message = "Conversion and denoising complete with errors."
            log_help = html.Div(
                [
                    html.H5(error_info, style={'textAlign': 'center'}),
                    html.Br(),
                    html.P(
                        "Please review the log file (log.log) to understand the error details, then adjust your configuration "
                        "and run the pipeline again.",
                        style={'textAlign': 'center'}
                    ),
                    html.Div(
                        dbc.Button(
                            "Open log file",
                            id='open-log-button',
                            color='danger',
                            n_clicks=0,
                        ),
                        style={'display': 'flex', 'justifyContent': 'center'}
                    ),
                    html.Br(),
                    html.Div(
                        html.Small(
                            "Once you have reviewed the log details, please retry the pipeline.",
                            style={'display': 'block', 'textAlign': 'center'}
                        )
                    ),
                    html.Div(id='log-open-feedback')
                ]
            )
            err = [separating_line, log_help]
            return err, progress, f"{progress}%" if progress > 0 else "", None, completion_message, True

progress = html.Div([
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    dcc.Interval(id="denoise-progress-interval", n_intervals=0, interval=500), # interval is the delay in ms when the data are updated
                    html.H5(children = 'Processing files, please wait... ', id = "conversion-info", style={'textAlign': 'center'}),
                    dbc.Progress(id="denoise-progress", animated=True, striped=True, style={"width": "600px"}),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                        ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',      # Center horizontally in the flex container
                        }),
                    html.Div(id = 'template-part')
                    ])


@callback(
    Output('log-open-feedback', 'children'),
    Input('open-log-button', 'n_clicks'),
    prevent_initial_call=True
)
def open_log_file(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    log_path = _session_log_path()

    try:
        if not log_path.exists():
            logging_config.configure_logging()
        _open_path_with_default_app(log_path.resolve())
        logging_config.log_info(logger, 'Log file opened from error prompt.')
        return dbc.Alert(
            [
                html.Span('Opening the session log. If it does not appear automatically, you can find it here: '),
                html.Code(str(log_path.resolve()))
            ],
            color='info',
            dismissable=True
        )
    except Exception as exc:
        logging_config.log_exception(logger, 'Unable to open the session log automatically.', exception=exc)
        return dbc.Alert(
            [
                html.Span('Unable to open the session log automatically. Please open it manually at: '),
                html.Code(str(log_path.resolve()))
            ],
            color='danger',
            dismissable=True
        )


# 8- Load sample template
###############################################################################
template_dict = {}
tables_list = []
label_tables_list = [] # to associate treatment class, corresponding labels to samples
treatment_groups = {}
experiment_titles = []

# Function to get the separator from csv file based on how much it is present in the file
def detect_delimiter(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        # Read the first 1024 bytes to get a sample of the file
        sample = file.read(1024)  # Count the occurrences of each delimiter
        comma_count = sample.count(',')
        semicolon_count = sample.count(';')
        
        # Determine the delimiter
        if comma_count >= semicolon_count:
            return ','
        else:
            return ';'

# Function to add a template to the treatment group
def add_template(template_path):
    global treatment_groups, current_project, tables_list, template_dict, label_tables_list, experiment_titles
    delimiter = detect_delimiter(template_path) # Detect the delimiter
    template = pd.read_csv(template_path, delimiter=delimiter)
    
    # Find ESI mode
    if len(template['Type_MS'].unique()) > 1:
        return False, f'{os.path.basename(template_path)} have at least two different types of MS mode, only one is possible. Verify Type_MS column values. {template["Type_MS"].unique()}'
    else:
        esi_mode = template['Type_MS'].unique()[0][-3:] #(pos or neg)
    if 'Verification' in template.columns:
        template = template.drop('Verification', axis=1)
    template = template.loc[template['Samples'].notna(),]
    # First verif if all samples names in the tempalte match samples from the mzml files list
    samples = template['Samples'].unique().tolist()
    mzml_files = [os.path.splitext(os.path.basename(file))[0] for file in current_project.mzml_files_path]
    for s in samples:
        if s not in mzml_files:
            return False, f'In {os.path.basename(template_path)}, {s} is not a sample listed in the mzml files from this project. Correct the name or remove it from this template.'
    columns_to_check = [col for col in template.columns if col[:10] == 'Treatment_']
    for c in columns_to_check:
        all_values_nothing = (template[c] == 'NOTHING').all()
        if not all_values_nothing and c in ['Treatment_1', 'Treatment_2', 'Treatment_3', 'Treatment4', 'Treatment_5', 'Treatment_6', 'Chem_treatment_1', 'Chem_treatment_2', 'Chem_treatment_3']:
            return False, f'In {os.path.basename(template_path)} a treatment col with treament values lacks a name. Use Treatment_yourclassname as column title for a class name.'
        
    columns_to_check += ['Technician', 'Investigator', 'Preparation_date', 'Line', 'Experiment_title']
    for c in columns_to_check:
        template_without_blank = template[~template[columns_to_check].apply(lambda x: x.isin(['BLANK']).any(), axis=1)] # Filter rows where any of the specified columns contain 'BLANK'
        all_values_nothing = (template_without_blank[c] == 'NOTHING').all()
        all_different = (len(template_without_blank[c].unique()) == len(template_without_blank['Samples'].unique()))
        if not all_values_nothing and not all_different:
            for label in template_without_blank[c].unique():
                samples = template_without_blank.loc[template_without_blank[c] == label, 'Samples'].to_list()
                if 'Treatment_' in c:
                    class_value = c.split('Treatment_')[1]
                elif 'Chem_treatment_' in c:
                    class_value = c.split('Chem_treatment_')[1]
                else:
                    class_value = c
                new_table = {
                    'label': label,
                    't_class' : class_value,
                    'samples': samples
                }
                label_tables_list.append(new_table)
                if (label, class_value) not in treatment_groups.keys():
                    treatment_groups[(label, class_value)] = samples
                    
                else: # 
                    for new_sample in samples:
                        if new_sample not in treatment_groups[(label, class_value)]: # in case of error
                            treatment_groups[(label, class_value)].append(new_sample)
    
    if template[columns_to_check].isin(['BLANK']).any().any():
        filtered_df = template[template[columns_to_check].apply(lambda x: x.isin(['BLANK']).any(), axis=1)] # Filter rows where any of the specified columns contain 'BLANK'
        blank_names = filtered_df['Samples'].unique().tolist() # Extract unique sample names
        filtered_df_without_blank = template[~template[columns_to_check].apply(lambda x: x.isin(['BLANK']).any(), axis=1)] # Filter rows where any of the specified columns contain 'BLANK'
        sample_n = filtered_df_without_blank['Samples'].unique().tolist() # Extract unique sample names
        
        exp_title = list(template['Experiment_title'].unique())[0]
        if exp_title in experiment_titles:
            exp_title += '_1'
        experiment_titles.append(exp_title)
        current_project.template_path[exp_title] = template_path
        current_project.template[exp_title] = template
        current_project.template_esi_mode[exp_title] = esi_mode
        tables_list.append({'exp_title' : exp_title, 'samples': sample_n, 'blanks':blank_names})
        template_dict[os.path.basename(template_path)] = template
        return True, ''
    else:
        return False, f'{os.path.basename(template_path)} lacks BLANK definition. Indicate BLANk in ont of the Treatment_ columns for blank run(s).' # if Blank lacking, return the path of the eincriminated tempalte

@callback(
    [Output('mzmine-part', 'children'),
     Output("template-name-output", "children"),
     Output("template-name-input", "valid"),
     Output("template-name-input", "invalid"),
     Output("template-name-input", "disabled"),
     Output("need-add-template-button", "disabled"),
     Output("meta-nomore-template-button", "disabled"),
     Output("template-name-input", "value"),
     Output({'type': 'popup', 'index': 6}, 'children')
     ],
    [Input("template-name-input", "n_submit"),
     Input("meta-nomore-template-button", "n_clicks"),
     Input("need-add-template-button", "n_clicks")],
    [State("template-name-input", "value")],
    prevent_initial_call = True
    )
def validate_template(n_submit_template, stop_add_template_clicks, add_template_clicks, template_path): 
    global template_dict
    global current_project, mzml_alternative, sample_names
    global label_tables_list, treatment_groups, line_count, tables_list
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    template_validity = False
    if button_id == 'template-name-input':
        template_path = normalize_path(template_path)
        if os.path.exists(template_path):              
            if os.path.isdir(template_path):
                if template_path in template_dict.keys():
                    remark = 'This template folder is already loaded.'
                    template_validity = False
                else:
                    search_pattern = os.path.join(template_path, '*.csv')
                    csv_files = glob.glob(search_pattern) # Use glob to find all files matching the pattern
                    for file in csv_files:
                        try:
                            template_validity, remark = add_template(file) # add the template(s) in tables_list
                        except Exception as e:
                            template_validity = False
                            remark = e
                            break
                        if template_validity == False:
                            break
            else:
                if template_path in template_dict.keys():
                    remark = 'This template is already loaded.'
                    template_validity = False
                else:
                    try:
                        template_validity, remark = add_template(template_path) # add the template(s) in tables_list
                    except Exception as e:
                        template_validity = False
                        remark = e

            if template_validity: # if oen or all tempalte are valid
                cache.set('project_loaded', current_project)
                
                return '', f'{os.path.basename(template_path)} succesfully loaded.', True, None, True, None, None, template_path, 'n'
            else:
                return '', remark, None, True, None, True, True, template_path, 'n'
        else:
            return '', 'File not found', None, True, None , True, True, template_path, 'n'
    elif button_id == "meta-nomore-template-button": # finished to enter sample templates
        to_delete = []
        for k, v in treatment_groups.items():
            if len(list(set(v))) == 1:
                to_delete.append(k) # del when there are unique values
                for t in label_tables_list:
                    label_tables_list = [table for table in label_tables_list if (k[0] != table['label'] and k[1] != table['t_class']) ]
        if label_tables_list == []:
            return 'Error', 'It seems in your Treatments or Line, there is only one label. No comparison is possible, review your template(s) and start again.', True, None, True, True, True, '', 'n'
        treatment_groups = {k: v for k, v in treatment_groups.items() if k not in to_delete}
        current_project.tables_list = tables_list
        current_project.label_tables_list = label_tables_list
        separating_line = create_separating_line(line_count)
        line_count += 1
        new_popup = html.Div(children = '', id={"type": "popup", "index": 7}, style={'display': 'none'})
        return [separating_line, new_popup, mzmine_settings], f'{os.path.basename(template_path)} succesfully loaded.', True, None, True, True, True, template_path, 'y'
    elif button_id == "need-add-template-button":
        
        return '', f'{os.path.basename(template_path)} succesfully loaded.', None, None, None, True, None, '', 'n'
    raise PreventUpdate()


template_part = html.Div([
                html.Div([
                    html.H5('Enter sample template or a folder path, and press Enter', style={'textAlign': 'center'}),
                    dbc.Input(id="template-name-input", valid = None, n_submit = 0, placeholder= "path/to/sample_template.csv", type="text", style={'maxWidth': '600px'}),
                    html.Br(),
                    html.P(id="template-name-output"),
                    html.Br(),
                    dbc.ButtonGroup([
                                dbc.Button('I need to add a template', id = "need-add-template-button", color="info", disabled = True, n_clicks = 0),
                                dbc.Button('I am done with sample template', id = "meta-nomore-template-button", color="primary", disabled = True, n_clicks = 0)
                                    ]),
                    html.Br(),
                    ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',      # Center horizontally in the flex container
                        }),
                html.Div(id = 'mzmine-part'),                
                ])          
                        
# 8- MZmine settings
###############################################################################  
# Global variables to track the threads for mzmine process
@callback(
    [Output("execution-part", "children"),
     Output("mzmine-software-msg", "children"),
     Output("batch-file-info", "children"),
     Output("one-feature-list", "disabled"),
     Output("meta-analysis", 'disabled'),
     Output("mzmine", "disabled"),
     Output('button-remark', "children"),
     Output({'type': 'popup', 'index': 7}, 'children')
     ],
    [Input("meta-nomore-template-button", "n_clicks"),
     Input("mzmine", "n_clicks"),
     Input("up-mzmine-button", "n_clicks"),
     Input("one-feature-list", "n_clicks"),
     Input("meta-analysis", 'n_clicks')],
    prevent_initial_call = True
)
def manage_batch(template_n_click, batch_n_clicks, n_clicks_mzmine, n_clicks_continue, n_clicks_all):
    global current_project, sample_names, mzml_alternative, first_process_thread, ms1_mzmine_instance, line_count, tables_list
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "mzmine":
        separating_line = create_separating_line(line_count)
        line_count += 1
        new_popup = html.Div(children = '', id={"type": "popup", "index": 8}, style={'display': 'none'})
        return [separating_line, new_popup, loading_mzmine], "", "Then modify batches by opening: /features/yourbatchfile.xml", True, True, True, '', 'y'
    
    elif button_id == "meta-analysis":
        if len(current_project.tables_list) == 1:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, dash.no_update, 'Meta analysis impossible!', 'n'
        Subfolder = current_project.featurepath
        current_project.meta = True
        for exp_title in current_project.template.keys():
            template = current_project.template[exp_title]
            folder = current_project.mzml_folder_path
            ext = '.mzML'
            file_list = [os.path.join(folder, s + ext) for s in template['Samples']]

            name = current_project.name + f'_{exp_title}_msn'
            batchname = os.path.join(current_project.featurepath, name + '.mzbatch')
            exportcsv = os.path.join(current_project.featurepath, name + '.csv')
            batch_default = './data/batch_default.mzbatch'
            shutil.copyfile(batch_default, batchname) # copy reference mzbatc h(an xml file) to the feature/ folder
            
            # add file names in the new batch file, and add also the export path
            replacement_text_files = "\n".join([f"            <file>{path}</file>" for path in file_list]) + "\n"
            replacement_text_featurelist = f"            <current_file>{exportcsv}</current_file>\n"
            replacement_text_featurelist += f"            <last_file>{exportcsv}</last_file>\n"
            with open(batchname, 'r') as file:
                lines = file.readlines()        
            start_tag_files = "<parameter name=\"File names\">"
            start_tag_featurelist = "<parameter name=\"Filename\">"
            end_tag = "</parameter>"        
            new_lines = []
            inside_target = False # to handle the end_tag which is not unique, to avoid matching the other end_tag "</parameter>"        
            for line in lines: # create the new file using a list of lines
                if start_tag_files in line:
                    new_lines.append(line)
                    new_lines.append(replacement_text_files)
                    inside_target = True
                elif start_tag_featurelist in line:
                    new_lines.append(line)
                    new_lines.append(replacement_text_featurelist)
                    inside_target = True
                elif inside_target and end_tag in line:
                    new_lines.append(line)
                    inside_target = False
                elif not inside_target:
                    new_lines.append(line)        
            with open(batchname, 'w') as file: # write the new file
                file.writelines(new_lines)
    
            current_project.featurelist[exp_title] = exportcsv
            current_project.batch[exp_title] = batchname

        cache.set('project_loaded', current_project)
        current_project.save()
        return "", "", "Then modify batches by opening: /features/yourbatchfile.xml", True, True, None, '', 'y'
    
    elif button_id == "one-feature-list":
        if len(current_project.template.keys()) > 1:
            exp_title = 'multi-template'
        else:
            exp_title = list(current_project.template.keys())[0]       

        file_list = current_project.mzml_files_path
        name = current_project.name + f'_{exp_title}_msn'
        batchname = os.path.join(current_project.featurepath, name + '.mzbatch')
        exportcsv = os.path.join(current_project.featurepath, name + '.csv')
        batch_default = './data/batch_default.mzbatch'
        shutil.copyfile(batch_default, batchname) # copy reference mzbatc h(an xml file) to the feature/ folder
        
        # add file names in the new batch file, and add also the export path
        replacement_text_files = "\n".join([f"            <file>{path}</file>" for path in file_list]) + "\n"
        replacement_text_featurelist = f"            <current_file>{exportcsv}</current_file>\n"
        replacement_text_featurelist += f"            <last_file>{exportcsv}</last_file>\n"
        with open(batchname, 'r') as file:
            lines = file.readlines()        
        start_tag_files = "<parameter name=\"File names\">"
        start_tag_featurelist = "<parameter name=\"Filename\">"
        end_tag = "</parameter>"        
        new_lines = []
        inside_target = False # to handle the end_tag which is not unique, to avoid matching the other end_tag "</parameter>"        
        for line in lines: # create the new file using a list of lines
            if start_tag_files in line:
                new_lines.append(line)
                new_lines.append(replacement_text_files)
                inside_target = True
            elif start_tag_featurelist in line:
                new_lines.append(line)
                new_lines.append(replacement_text_featurelist)
                inside_target = True
            elif inside_target and end_tag in line:
                new_lines.append(line)
                inside_target = False
            elif not inside_target:
                new_lines.append(line)        
        with open(batchname, 'w') as file: # write the new file
            file.writelines(new_lines)

        current_project.featurelist[exp_title] = exportcsv
        current_project.batch[exp_title] = batchname
        cache.set('project_loaded', current_project)
        current_project.save()
        return "", "", "Then modify batches by opening: /features/yourbatchfile.xml", True, True, None, '', 'y'
    
    elif button_id == "up-mzmine-button":
        try:
            mzmine_path = _get_configured_software_path('MZmine', 'MZmine 3 (MZmine.exe)')
        except SoftwarePathError as exc:
            logging_config.log_warning(logger, str(exc))
            warning = dbc.ListGroupItem(str(exc), color="warning", style=SOFTWARE_WARNING_STYLE)
            return "", warning, "Then modify batches by opening: /features/yourbatchfile.xml", dash.no_update, dash.no_update, dash.no_update, '', 'n'

        try:
            _launch_external_software(mzmine_path)
            return "", "", "Then, open the feature list batch file from /features/yourbatchfile.xml", dash.no_update, dash.no_update, dash.no_update, '', 'n'
        except FileNotFoundError as exc:
            logging_config.log_error(logger, 'MZmine executable not found: %s', exc)
            error = dbc.ListGroupItem(
                "MZmine executable cannot be found at the configured location. Please verify the path from the Login page.",
                color="danger",
                style=SOFTWARE_WARNING_STYLE,
            )
            return "", error, "Then modify batches by opening: /features/yourbatchfile.xml", dash.no_update, dash.no_update, dash.no_update, '', 'n'
        except OSError as exc:  # pragma: no cover - defensive
            logging_config.log_error(logger, 'Failed to open MZmine: %s', exc, exc_info=True)
            return "", "Failed to open MZmine 3. Try to open it externally.", "Then modify batches by opening: /features/yourbatchfile.xml", dash.no_update, dash.no_update, dash.no_update, '', 'n'

    else:
        raise PreventUpdate()
    
mzmine_settings = html.Div([
                    html.Div([
                        dbc.ListGroupItem("""For the next step, we will call an open source software called MZmine. This software is a 100% tunable device to filter the mass spectra and generate a feature list. \
            A feature is a signal, which can be a compound ion or noise, detected by the mass spectrometer and which have a shape recognized by the software depending on the defined settings. As we said, feature is not necessarly a compound, it can be instrument noise, it can be a chemical contaminant, it can be anything detected. \
            The feature list are all the relevant signals computationally dectected by MZmine and which could potentially be an existing compound. Depending on the LC properties, a feature from an existing compound look like a gaussian curve. Each feature has properties like a retention time and a m/z, and is found present or not in your samples. \
            Now, we need to generate batch files, in .xml format. A batch file is a file containing instructions for MZmine to generate a feature list. It contains all the parameters defined by default to identify the features.
                
            The advantage of MZmine is that hundreds of parameters can be defined to generate the feature list.""", color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px',}),
                        
                        html.Br(), 
                        dbc.Alert([
                            html.Div([
                                html.H6("Optional", style={'margin-top': '0px', 'margin-bottom': '5px'}),
                                html.P("To modify MZmine settings, clik on the button below to go on MZmine > Project > Batch mode.", style={'margin-top': '0px', 'margin-bottom': '0px'}),
                                html.P("Then, open the your feature list batch file from Your_project/features/your_featurelist_name_ms1(ms2).xml", id = 'batch-file-info', style={'margin-top': '0px', 'margin-bottom': '0px'})
                                    ], style={
                                            'display': 'flex',
                                            'flexDirection': 'column',
                                            'alignItems': 'center',      # Center horizontally in the flex container
                                        }),
                                html.Hr(style={'margin-top': '10px', 'margin-bottom': '10px'}),
                                html.Div([
                                    html.Button(
                                        children=[html.Img(src='/assets/mzmine_logo.png', style={'width': 'auto', 'height': '60px'})],
                                        id='up-mzmine-button',
                                        style={'background': 'none', 'border': 'none', 'padding': '0', 'margin-top': '0px', 'margin-bottom': '0px'},
                                        n_clicks=0),
                                        ], style={
                                                'display': 'flex',
                                                'flexDirection': 'column',
                                                'alignItems': 'center',      # Center horizontally in the flex container
                                            }),
                                    ], color="info", style={'padding-top': '0.5rem', 
                                                            'padding-bottom': '0.5rem',
                                                            'padding-right': '2rem', 
                                                            'padding-left': '2rem', 
                                                            'width' : '600px'}),                        
                        html.P(id="mzmine-software-msg"),
                        html.Br(),
                        html.P('Only if you run in meta-analysis mode:'),
                        html.Div([
                        dbc.InputGroup(
                            [dbc.InputGroupText("RT +-"),
                                dbc.Input(id='rt-threshold-templates', type="number", value=0.03, step=0.01),
                                dbc.InputGroupText("min")],
                                className="me-1",
                                size="sm"),
                        dbc.Tooltip(
                            "RT difference for matching features of a same m/z between two templates",
                            target="rt-threshold",  # ID of the component to which the tooltip is attached
                            placement="left"),
                        dbc.InputGroup(
                            [dbc.InputGroupText("Spearman correlation"),
                                dbc.Input(id='corr-threshold-templates', type="number", value=0.9, step=0.01, min=0, max=1)],
                                className="me-1",
                                size="sm"),
                        dbc.Tooltip(
                            "m/z difference between two features in two templates in a RT threshold to be considered the same",
                            target="mz-threshold",  # ID of the component to which the tooltip is attached
                            placement="left"),
                                ], style={'display': 'flex','margin-bottom': '10px'}),
                        html.Br(),
                        dbc.ButtonGroup([
                            dbc.Spinner(dbc.Button('One feature list', id = "one-feature-list", color="danger", n_clicks=0)),
                            dbc.Spinner(dbc.Button('Meta-analysis', id = "meta-analysis", color="warning", n_clicks=0)),      
                            ]),
                        html.Div(id = 'button-remark', children = ''),
        
                        dbc.Tooltip(
                            "One mzmine run which combine all samples from all templates in one feature list. If more than one template, use it only when the experiments and LCMS settings are rigorously identical.",
                            target="one-feature-list",  # ID of the component to which the tooltip is attached
                            placement="left"),
                        dbc.Tooltip(
                            "For meta-analysis with many templates, involving experiments with slight differences.",
                            target="meta-analysis",  # ID of the component to which the tooltip is attached
                            placement="top"),
                        html.Br(),
                        dbc.Alert(
                            [
                                html.P(
                                    "Before launching MZmine, make sure the application is running and that a valid account is signed in. "
                                    "If no account is logged in, MZmine will not generate the feature list.",
                                    style={'margin-bottom': '0px'}
                                )
                            ],
                            color="warning",
                            style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px', 'padding-right': '5px'}
                        ),
                        html.Br(),
                        dbc.Button('Launch MZmine', id='mzmine', color="primary", n_clicks=0, disabled = True),
                        dbc.Tooltip(
                            "The batch files are generated in /feature. You can open and modify them with MZmine",
                            target="rt-threshold",  # ID of the component to which the tooltip is attached
                            placement="left"),
                        html.Br(),
                        ], style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'alignItems': 'center',      # Center horizontally in the flex container
                            }),
                    html.Div(id = 'execution-part'),  
                    ])

# 9- Mzmine execution
###############################################################################
mzmine_process_start = False
template_to_proceed = []
exp_title_to_proceed = None
process_thread = None
mzmine_instance = None
message = None
status = None
button_status = None

@callback(
    [Output('mzmine-process-status', 'children'),
    Output('interval-component', 'disabled'),
    Output("continue-deblanking-button", 'disabled')],
    [Input("continue-deblanking-button", 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    Input("mzmine", 'n_clicks')]    
         )
def update_process_status(n_clicks_deblank, n, n_clicks):
    global process_thread, mzmine_instance, template_to_proceed, nb_of_templates, exp_title_to_proceed, mzmine_process_start
    global message, status, button_status
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'continue-deblanking-button':
        return "All templates done", True, True

    
    
    
    if not mzmine_process_start:
        mzmine_process_start = True # to handle the templates to proceed
        template_to_proceed = list(current_project.batch.keys()) # list of exp titles
        nb_of_templates = len(template_to_proceed)
        
    if len(template_to_proceed) != 0:   
        exp_title_to_proceed = template_to_proceed[0]        
        progress_template = f' ({nb_of_templates - len(template_to_proceed) + 1}/{nb_of_templates})...'
    else:
        return "All templates done", True, None
    
    if process_thread is None:
        # Initialize MZmine object
        fl_msn_path = current_project.featurelist[exp_title_to_proceed]
        batch_path = current_project.batch[exp_title_to_proceed]
        mzmine_instance = features.MZmine(batch_path, fl_msn_path) # run mzmine for ms1
        process_thread = mzmine_instance.start_run()
        return [dbc.Spinner(color="success", type="grow", spinner_style={'marginBottom': '30px'}), html.P(exp_title_to_proceed + progress_template)], None, True
    
    elif process_thread is not None :
        if not process_thread.is_alive():
            if mzmine_instance:
                if 'Feature' in mzmine_instance.message: # stands for good log output f'Feature list {os.path.basename(self.fln)} created.'
                    mzmine_instance = None
                    message =  f"{exp_title_to_proceed} Done"
                    status = None
                    del template_to_proceed[0]
                    process_thread = None
                    return message, status, True
                else:
                    message =  mzmine_instance.message
                    status = True
                    return message, status, True
            else:
                return message, status, True
        else:
            return dash.no_update, None, True
    return [dbc.Spinner(color="success", type="grow", spinner_style={'marginBottom': '30px'}), html.P(exp_title_to_proceed + progress_template)], None, True

mzmine_msn = html.Div([
    html.Div("MZmine processing:  ", style={'display': 'inline-flex', 'alignItems': 'center', 'marginRight': '10px'}),
    html.Div(id="mzmine-process-status", children=[dbc.Spinner(color="success", type="grow", spinner_style ={'marginBottom': '30px'})], 
             style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center'}),
        ], style={'display': 'flex', 'justifyContent': 'start', 'alignItems': 'center'})


loading_mzmine = html.Div([
                    html.Div([
                            dcc.Store(id='loading-state', data={'loading': False}),
                            mzmine_msn,
                            html.Br(),
                            html.Br(),                            
                            dcc.Interval(id='interval-component', interval=1000, n_intervals=0, max_intervals=-1),  # Checks every second
                            dcc.Interval(id='continue-interval-component', interval=1000, n_intervals=0, max_intervals=-1),
                            html.P("Define the deblanking ratio:"),
                            html.Div([
                            dbc.InputGroup([
                                dbc.InputGroupText("Ratio: 1/"),
                                dbc.Input(id = 'deblanking-level', value=5, type="number", min=2, max=20, step=1),
                                dbc.Button('Remove blank signal', id = "continue-deblanking-button", color="primary", n_clicks=0, disabled = True),
                                ]),
                                ], style={'display': 'flex', 'maxWidth': '600px', 'alignItems': 'center'}),
                            dbc.Tooltip(
                                "For example a ratio of 5 means for a given feature, a blank signal higher than 1/5 of the signal in a given sample will delete the signal from this sample (high probability of blank feature, not sample feature).",
                                target="'deblanking-level",  # ID of the component to which the tooltip is attached
                                placement="left"),
                            html.Div(id = 'template_deblanking', children=['']),
                            html.Br(),
                            html.Br(),
                            
                            ], 
                          style={
                                'display': 'flex',
                                'flexDirection': 'column',
                                'alignItems': 'center',      # Center horizontally in the flex container
                            }),
                    html.Div(children = '', id = 'return-main-part')
                    ])


# 10 - Deblanking and labelling part
###############################################################################
feature_id = 0

# Function to transform mzmine csv output in something more summarized and readable
def adjust_process(Fl_path):  # JUST FOR A SINGLE MS LEVEL
    global current_project
    global feature_id

    project_context = current_project if getattr(current_project, "update_log", None) else None

    logging_config.log_info(
        logger,
        "Loading feature list from %s",
        Fl_path,
        project=project_context,
    )

    try:
        featurelist_adj = pd.read_csv(Fl_path) # msx stands for ms1 or ms2
    except FileNotFoundError as exc:
        logging_config.log_exception(
            logger,
            "Feature list missing at %s",
            Fl_path,
            project=project_context,
            exception=exc,
        )
        raise
    except Exception as exc:
        logging_config.log_exception(
            logger,
            "Unable to load feature list %s",
            Fl_path,
            project=project_context,
            exception=exc,
        )
        raise

    # Extract sample names and create a mapping dictionary for renaming columns
    column_mapping = {}
    
    for col in featurelist_adj.columns:
        if col.startswith("datafile:"):
            parts = col.split(':')
            sample_name = parts[1]
            measurement = parts[2]
            
            # Construct new column names based on the specific measurement type
            if measurement == 'mz':
                new_name = f'{sample_name} Peak m/z'
            elif measurement == 'rt':
                new_name = f'{sample_name} Peak RT'
            elif measurement == 'rt_range':
                if parts[3] == 'min':
                    new_name = f'{sample_name} Peak RT start'
                elif parts[3] == 'max':
                    new_name = f'{sample_name} Peak RT end'
            elif measurement == 'intensity_range' and parts[3] == 'max':
                new_name = f'{sample_name} Peak height'
            elif measurement == 'area':
                new_name = f'{sample_name} Peak area'
            elif measurement == 'mz_range':
                if parts[3] == 'min':
                    new_name = f'{sample_name} Peak m/z min'
                elif parts[3] == 'max':
                    new_name = f'{sample_name} Peak m/z max'
            else:
                new_name = col
            # Add to dictionary
            column_mapping[col] = new_name
        else:  # Preserving some columns explicitly
            column_mapping[col] = col

    featurelist_adj_renamed = featurelist_adj.rename(columns=column_mapping) # Rename columns using the mapping
    base_columns = ['rt', 'mz']

    sample_columns = set()
    for col in column_mapping.values(): # Add sample-specific columns based on the transformed name
        if any(prefix in col for prefix in ['Peak m/z', 'Peak RT', 'Peak area', 'Peak height', 'Peak RT start', 'Peak RT end', 'Peak m/z min', 'Peak m/z max']):
            sample_columns.add(col)
    columns_to_keep = base_columns + list(sample_columns) # Final set of columns to retain in the DataFrame
    featurelist_adj_final = featurelist_adj_renamed[columns_to_keep] # Filter the renamed DataFrame to only include these columns
   
    sample_columns = {}
    for col in featurelist_adj_final.columns: # Identify columns for each sample and the pattern for the column names
        if "Peak" in col:
            sample_name = col.split(" Peak")[0]
            if sample_name not in sample_columns:
                sample_columns[sample_name] = []
            sample_columns[sample_name].append(col)
    
    rows_list = [] # Prepare to transform the DataFrame
    for index, row in featurelist_adj_final.iterrows(): # Iterate over each row in the dataframe
        feature_rt = row['rt']
        feature_mz = row['mz']
        
        # For each sample, check if there is a signal (non-null Peak area)
        for sample, cols in sample_columns.items():
            if not pd.isna(row[sample + " Peak area"]):  # Check if 'Peak area' is non-null
                new_row = {
                    'feature': index + feature_id + 1, 
                    'rt': feature_rt,
                    'm/z': feature_mz,
                    'sample': sample,
                    'peak_mz': row[sample + " Peak m/z"],
                    'peak_rt': row[sample + " Peak RT"],
                    'area': row[sample + " Peak area"],
                    'height': row[sample + " Peak height"],
                    'peak_rt_start': row[sample + " Peak RT start"],
                    'peak_rt_end': row[sample + " Peak RT end"],
                    'peak_mz_min': row[sample + " Peak m/z min"],
                    'peak_mz_max': row[sample + " Peak m/z max"]
                }
                rows_list.append(new_row)
    
    # Create the new DataFrame from rows_list
    featurelist_adj_final = pd.DataFrame(rows_list)

    if featurelist_adj_final.empty:
        logging_config.log_warning(
            logger,
            "Feature list %s produced no usable rows after processing.",
            Fl_path,
            project=project_context,
        )
        return featurelist_adj_final

    if 'sample' in featurelist_adj_final.columns:
        featurelist_adj_final['sample'] = featurelist_adj_final['sample'].str.replace('.mzML', '', regex=False)

    feature_id = int(featurelist_adj_final['feature'].max())

    logging_config.log_info(
        logger,
        "Loaded %d processed features from %s",
        len(featurelist_adj_final),
        Fl_path,
        project=project_context,
    )
    return featurelist_adj_final

def match_in_all_sub(row, all_sub_df):
    sample_name = row['sample']
    # Check if sample_name is a column in all_sub_df
    if sample_name in all_sub_df.columns:
        # Filter all_sub_df for rows where m/z and rt match
        filtered_df = all_sub_df[(all_sub_df['m/z'] == row['m/z']) & (all_sub_df['rt'] == row['rt'])]
        # If any row is found, the feature is present for this sample in all_sub
        return not filtered_df.empty
    return False

import collections
def _group_size_counts(graph):
    sizes = [len(c) for c in nx.connected_components(graph)]
    cnt = collections.Counter(sizes)  # {size: how_many_groups}
    return len(sizes), cnt

# For meta analysis, we make a feature grouping per tempalte, then inter tempaltes
def feature_grouping(Msn_deblanked, Rt_threshold, Correlation_threshold):
    global current_project, deblanking_featuregroup_info
    minimum_ratio_threshold = 1 - (5/len(current_project.sample_names)) # 0.5 for 10 samples, it is nb of column (samples) with non-0 values at least to be shared between two features to validate a correlation. The more sampels in an experiment, the higher this threshold to avoid chained wrong corrrelation 
    
    feature_id = 1 # a unique id number is given to all features from all experiments
    ID = 1 # correspond to group node (feature group) IDs
    experiments = {} # keys: sample temaltes, values: a dict with key FG: {'identifier': ID, 'subset' : stand_subset_df, 'avg_row': stand_avg_row}, key ion_df: stand_ion_df for the whole template
    experiment_FG = {}
    # let begin by making feature groups per tempalte, following same principles as in analytics/ for single templates
    for table in current_project.tables_list:        
        exp_title = table['exp_title']        
        esi_mode = current_project.template_esi_mode[exp_title]
        deblanking_featuregroup_info = f'Meta-analysis: feature grouping of template {exp_title}...'
        samples = table['samples']
        experiments[exp_title] = {'samples' : table['samples']} # take each template, take the list of samples
        exp_fl = Msn_deblanked.loc[Msn_deblanked['sample'].isin(samples), ] # experiment fl, meaning all the rows from Msn_deblanked which belog to this experiment
        # Prepare the extracted feature list
        exp_fl_pivot_df = exp_fl.pivot_table(index='feature', columns='sample', values='area', fill_value=0) # take feature as rows, area as values, sampels as columns
        rt_mz_df = exp_fl[['feature', 'rt', 'm/z']].drop_duplicates().set_index('feature')  # take rt and mz col
        exp_absolute_preprocessed_df = exp_fl_pivot_df.merge(rt_mz_df, left_index=True, right_index=True) # combine the last two df in one
        exp_stand_preprocessed_df = exp_absolute_preprocessed_df.drop(columns = ['rt', 'm/z'])
        exp_stand_preprocessed_df = exp_stand_preprocessed_df.div(exp_stand_preprocessed_df.max(axis=1), axis=0)
        exp_stand_preprocessed_df['rt'] = exp_absolute_preprocessed_df['rt']
        exp_stand_preprocessed_df['m/z'] = exp_absolute_preprocessed_df['m/z']
        
        # Make the correlation
        #######################################################################
        G = nx.Graph()   # Create the G network, meaning the python object for feature network

        # Group features based on similar retention time
        # For each feature, find other features within the RT threshold
        similar_rt_groups = {}
        for feature in exp_stand_preprocessed_df.index:
            rt = exp_stand_preprocessed_df.loc[feature, 'rt']
            similar_features = exp_stand_preprocessed_df[np.abs(exp_stand_preprocessed_df['rt'] - rt) <= Rt_threshold].index.tolist()
            similar_rt_groups[feature] = similar_features
            
        # Adjsut the similar rt grouping, to keep only the the group with high sized, to avoid a fature to make a false positive correlation between two relevant feature groups
        unique_lists = list({tuple(v) for v in similar_rt_groups.values()})
        unique_lists = [list(t) for t in unique_lists]
        similar_rt_groups_adjusted = {}
        for feature in exp_stand_preprocessed_df.index:
            similar_rt_groups_adjusted[feature] = max((lst for lst in unique_lists if feature in lst), key=len, default=[]) # take only the biggest clusters of rt for a feature, to avoid that in-between feature groupfeature goes for false correlation of feature group
            
        experiments[exp_title]['similar_rt_groups'] = similar_rt_groups_adjusted
        all_pairs_possible = [(feature, other_feature) for feature, group in similar_rt_groups_adjusted.items() for other_feature in group if feature < other_feature]
        pairs_df = pd.DataFrame(all_pairs_possible)
        pairs_df.columns = ['feature', 'other_feature']
        rt_values = exp_stand_preprocessed_df.loc[:,'rt']
        pairs_time_difference = np.abs(rt_values.values[:, None] - rt_values.values[None, :]) # absolute tome difference between eahc pair of feature
        pairs_time_difference = pd.DataFrame(pairs_time_difference, index=rt_values.index, columns=rt_values.index)
        
        # Replace 0s with NaNs in valid_rows
        valid_rows_df = exp_stand_preprocessed_df.iloc[:,:-2].replace(0, np.nan)

        # Compute the correlation matrix, ignoring NaN values
        correlation_matrix_sp = valid_rows_df.transpose().corr(method='spearman').round(2)
        
        # Compute the interesected matrix, for the minimal shared column threshold. The idea is to have how many non-0 col in both pair of features are shared, for the correlation consistency
        non_nan_mask = valid_rows_df.notna().astype(int)
        shared_non_nan = np.dot(non_nan_mask.values, non_nan_mask.values.T) # Use broadcasting to create a 3D array where each (i, j) element is the AND operation between row i and row j
        total_non_nan = (non_nan_mask.values[:, None, :] | non_nan_mask.values[None, :, :]).sum(axis=2)# Calculate the total non-NaN values for each pair of rows using logical OR 
        ratio_matrix  = shared_non_nan / total_non_nan # for each pair of feature, tell the ratio of shared non-0 column 
        ratio_matrix = pd.DataFrame(ratio_matrix, index=valid_rows_df.index, columns=valid_rows_df.index)
        
        # Create the edges list
        edges = [] # edges list tuples of len 3, with two correlated feature IDs and their corresponding correlation
        # Iterate over the pairs in pairs_df
        for _, row in pairs_df.iterrows():
            feature = row['feature']
            other_feature = row['other_feature']
            
            # Check if the correlation exists and is greater than 0.8
            if feature in correlation_matrix_sp.index and other_feature in correlation_matrix_sp.columns:
                corr_value = correlation_matrix_sp.loc[feature, other_feature]
                ratio_value = ratio_matrix.loc[feature, other_feature]
                rt_diff = pairs_time_difference.loc[feature, other_feature]
                if corr_value >= Correlation_threshold and ratio_value >= minimum_ratio_threshold and rt_diff <= Rt_threshold:
                    edges.append((feature, other_feature, corr_value))
        
        for feature in exp_stand_preprocessed_df.index:
            G.add_node(feature, mz=exp_stand_preprocessed_df.loc[feature, 'm/z'])
        for node in G.nodes():
            G.nodes[node]['mz'] = exp_stand_preprocessed_df.loc[node, 'm/z']
            G.nodes[node]['rt'] = exp_stand_preprocessed_df.loc[node, 'rt']
            
        # Add edges with correlation as edge attribute
        G.add_weighted_edges_from(edges)
        
        G_before = G.copy()
        total_before, counts_before = _group_size_counts(G_before)
        
        # Apply the Louvain method for community detection. in order to better separated feature groups (mostly due wrong association of two nodes between two bigger groups)
        communities = nx_comm.louvain_communities(G, weight='weight', resolution= 1)

        # Create a mapping of nodes to their community
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i

        # Identify and remove edges between communities
        edges_to_remove = []
        for u, v in G.edges():
            if community_map[u] != community_map[v]:
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)
        
        G_after = G.copy()
        G_after.remove_edges_from(edges_to_remove)
        total_after, counts_after = _group_size_counts(G_after)
        
        # ----- BUILD SUMMARY TABLE -----
        all_sizes = sorted(set(counts_before.keys()) | set(counts_after.keys()))
        row_before = {"state": "before", "total_groups": total_before}
        row_after  = {"state": "after",  "total_groups": total_after}
        
        for s in all_sizes:
            row_before[f"rank{s}"] = counts_before.get(s, 0)
            row_after[f"rank{s}"]  = counts_after.get(s, 0)
        
        df_summary = pd.DataFrame([row_before, row_after])
        
        # drop rank columns that are zero in BOTH rows
        rank_cols = [c for c in df_summary.columns if c.startswith("rank")]
        zero_both = [c for c in rank_cols if df_summary[c].sum() == 0]
        df_summary = df_summary.drop(columns=zero_both)
        
        # write CSV (adjust path if you have an export folder)
        out_csv = os.path.join(".", f"group_size_summary_{exp_title}.csv")  # e.g., "./group_size_summary_MyExp.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[MassLearn] Group-size summary written to: {out_csv}")

    
        # Creating a new DataFrame with the same structure as p_df but empty
        stand_ion_df = pd.DataFrame(columns = exp_stand_preprocessed_df.columns) # ion_df will be all the unique precusor ion, with highest m/z as default representing m/z. Each line of ion_df represent a precusor ion, which is composed of one or multiple features
        experiments[exp_title]['FG'] = {}
        
        #neutral_mass_df = {'neutral_mass':[], 'rt':[], 'adducts':[], 'Feature_id' : [], 'FG':[], 'neutral_identity_group':[]} # df containg all neutral masses from the template run
        for component in nx.connected_components(G): # component is a set of ions   
            experiment_FG[ID] = {'experiment': exp_title}               
            absolute_subset_df = exp_absolute_preprocessed_df[exp_absolute_preprocessed_df.index.isin(component)] # on standardized values of the features, to determinate fragment ratios

            # take standardized value for the feature group to create at the end the stand ion df of the given template
            stand_subset_df = exp_stand_preprocessed_df[exp_stand_preprocessed_df.index.isin(component)]
            max_mass = stand_subset_df['m/z'].max()
            mean_rt = stand_subset_df['rt'].mean()
            stand_avg_row = stand_subset_df.drop(columns=['m/z', 'rt'])
        
            # Calculate the average of the filtered rows (excluding the 'feature' column)
            stand_avg_row = stand_avg_row.mean(numeric_only=True).to_frame().transpose()
            stand_avg_row['m/z'] = max_mass # highest mass is the reference mass, because it is more likely to be the closet to the precusor ion mass
            stand_avg_row['rt'] = mean_rt
            stand_avg_row['Size'] = len(stand_subset_df) # number of features present
            stand_avg_row['FG'] = ID
            
            # Now perform the concatenation
            stand_ion_df = pd.concat([stand_ion_df, stand_avg_row], ignore_index=True)
        
            # Update experiments templates associated data dictionnary
            stand_subset_df['FG'] = ID # add a column for feature group id in subset df which contain all features from the feature group
            
            # Function to check if two rows are within the specified range, to avoid very similar masses in the same subset
            def is_within_range(row1, row2):
                return abs(row1['m/z'] - row2['m/z']) <= 0.0015
            
            unique_combinations = list(combinations(range(len(stand_subset_df)), 2))
            rows_to_del = [] # keep track of rows to delete
            for i, j in unique_combinations:
                if is_within_range(stand_subset_df.iloc[i], stand_subset_df.iloc[j]):
                    rows_to_del.append(j)
            rows_to_del = list(set(rows_to_del)) # keep only unique values
            stand_subset_df_reduced = stand_subset_df.drop(stand_subset_df.index[rows_to_del]) # Create a new dataframe with only the rows to be kept
            
            # Create a df where there is the potential mass spectra of for a feature group
            relative_subset_df = absolute_subset_df.drop(columns=['m/z', 'rt']) # take only the samples
            relative_subset_df = relative_subset_df.drop(stand_subset_df.index[rows_to_del])
            relative_subset_df = relative_subset_df.apply(lambda x: x / x.max(), axis=0) # standardize the mass spectra per sample
            stand_mass_spectra = relative_subset_df[relative_subset_df != 0].mean(axis=1)
            lowest_relative = relative_subset_df.apply(lambda row: row.min(), axis=1)
            highest_relative = relative_subset_df.apply(lambda row: row.max(), axis=1)
            relative_subset_df['stand_mass_spectra'] = stand_mass_spectra
            relative_subset_df['lowest_relative'] = lowest_relative
            relative_subset_df['highest_relative'] = highest_relative
            
            # Fg mass spectra df have the list of all features (m/z) in a feature group as rows, the col stand_mass_spectra gives the relative standardized intensity of a m/z compared to the highest m/z present in the feature group, lowest and highest relative are the interval where we found the intensities values for each feature
            fg_mass_spectra = pd.concat([relative_subset_df[['stand_mass_spectra', 'lowest_relative', 'highest_relative']], stand_subset_df_reduced[['m/z', 'rt']]], axis=1)
            
            # Finding potential neutral mass from adducts
            original_masses = stand_subset_df_reduced['m/z'].to_list()
            original_rt = stand_subset_df_reduced['rt'].to_list()
            if esi_mode == 'pos':
                adducts = pd.read_csv('./data/positive_mode_adducts.csv')
            elif esi_mode == 'neg':
                adducts = pd.read_csv('./data/negative_mode_adducts.csv')
            neutral_masses_possibilities_df = {} # take original masses deteted in the feautre list as keys and their potential neutral masses based on most common adducts as values (list)
            neutral_masses_possibilities_dict = {}
            feature_id_list = []
            for m, r in zip(original_masses, original_rt):
                neutral_masses_possibilities_df[m] = [m - m_c for m_c in adducts['mass_change']]
                for add, m_c in zip(adducts['adduct'], adducts['mass_change']) :
                   neutral_masses_possibilities_dict[m + m_c] = [exp_title, ID, feature_id, m, r, add]
                feature_id_list.append(feature_id)
                feature_id += 1 # feature id is am original mass from the feature list
            fg_mass_spectra.index = feature_id_list
            fg_mass_spectra = fg_mass_spectra.rename_axis('feature_id_meta')
            experiment_FG[ID]['fg_mass_spectra'] = fg_mass_spectra
            experiments[exp_title]['FG'][ID] = {'stand_subset' : stand_subset_df, 'stand_avg_row': stand_avg_row, 'absolute_subset_df' : absolute_subset_df, 'fg_mass_spectra': fg_mass_spectra} # add the subset of features for the corresponding FG, and its associated average value row

            neutral_masses_possibilities_df = pd.DataFrame(neutral_masses_possibilities_df, index = adducts['adduct']).T # the original masses are rows and adducts are columns
            
            experiments[exp_title]['FG'][ID]['neutral_masses_possibilities_dict'] =  neutral_masses_possibilities_dict
            experiment_FG[ID]['neutral_masses_possibilities_dict'] = neutral_masses_possibilities_dict
            ID += 1 
        experiments[exp_title]['ion_df'] = stand_ion_df # add for the given experiment its ion_df containing all FGs
        #experiments[exp_title]['neutral_mass_df'] = pd.DataFrame(neutral_mass_df)
    current_project.feature_id = feature_id - 1
    current_project.experiments = experiments
    current_project.experiment_FG = experiment_FG

deblanking_featuregroup_info = '' # global variable to display remove blank signal information
@callback(
    [Output('return-main-part', 'children'),
     Output('continue-deblanking-button', 'children'),
     Output('continue-interval-component', 'disabled'),
     Output('template_deblanking', 'children')],
    [Input('continue-deblanking-button', 'n_clicks'),
     Input('continue-interval-component', 'n_intervals'),
     State('return-main-part', 'children')],
    prevent_initial_call = True
         )
def loading_button(n_clicks, continue_interval, return_children):
    global deblanking_featuregroup_info, line_count
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'continue-deblanking-button':
        return dash.no_update, [dbc.Spinner(size="sm"), " Wait for deblanking..."], None, deblanking_featuregroup_info
    elif button_id == 'continue-interval-component':
        if current_project.complete == True:
            separating_line = create_separating_line(line_count)
            line_count += 1
            return [separating_line, return_main], 'Blank signal removed', True, deblanking_featuregroup_info
        else:
            return dash.no_update, dash.no_update, None, deblanking_featuregroup_info
    raise PreventUpdate()

def deblank_and_grouping(Level, Rt_threshold, Correlation_threshold):
    global current_project, line_count, line_count, treatment_groups, deblanking_featuregroup_info
    deblanking_featuregroup_info = 'Removing blank from feature list(s)...'
    msn_df_list = []
    project_context = current_project if getattr(current_project, "update_log", None) else None
    project_name = getattr(current_project, "name", "unknown")

    logging_config.log_info(
        logger,
        "Starting deblanking for project %s (blank ratio 1/%s, rt threshold %s, correlation threshold %s)",
        project_name,
        Level,
        Rt_threshold,
        Correlation_threshold,
        project=project_context,
    )
    for exp_title in current_project.featurelist.keys():
        fl_msn_path = current_project.featurelist[exp_title]
        logging_config.log_info(
            logger,
            "Processing feature list '%s' for project %s",
            fl_msn_path,
            project_name,
            project=project_context,
        )
        msn_adjusted = adjust_process(fl_msn_path)
        msn_df_list.append(msn_adjusted)
    msn_df = pd.concat(msn_df_list, ignore_index=True)
    if msn_df.empty:
        logging_config.log_warning(
            logger,
            "Combined feature list for project %s is empty after processing.",
            project_name,
            project=project_context,
        )
    current_project.msn_df = msn_df
    msn_df_path = os.path.join(current_project.featurepath, current_project.name + '_msn.csv')
    current_project.msn_df_path = msn_df_path
    try:
        current_project.msn_df.reset_index(drop=True).to_csv(msn_df_path, index=False)
    except OSError as exc:
        logging_config.log_exception(
            logger,
            "Failed to write combined feature list to %s",
            msn_df_path,
            project=project_context,
            exception=exc,
        )
        raise

    # Deblanking part
    tables_list = current_project.tables_list
    msn_area = msn_df.pivot_table(index=['m/z', 'rt'], columns='sample', values='area').fillna(0)
    sub_tables = []    
    for t in tables_list:
        blank_list = t['blanks']
        sample_list = t['samples']            
        existing_blank_columns = [col for col in blank_list if col in msn_area.columns] # Ensure that the columns in blank_list exist in msn_area, because it can happens there is no signal at all detected in blanks and they are filtered out of msn_df
        if existing_blank_columns:
            table_blank = msn_area[existing_blank_columns]
        else:
            table_blank = pd.DataFrame(columns=msn_area.columns, index=msn_area.index).iloc[:, :0]
        table_blank = table_blank.replace(0, np.nan)
        table_blank['mean_blank'] =  table_blank.mean(axis=1, skipna=True) 
        table_sample = msn_area[sample_list]
        condition = table_sample <= Level * table_blank['mean_blank'].values[:, np.newaxis] 
        table_sample = table_sample.mask(condition, 0) # Apply the condition to filter and update values
        sub_tables.append(table_sample)
    all_sub_tables = pd.concat(sub_tables, axis=1)    # concatenate all the tables with samples which were separated due to different blanks values
    all_sub_tables = all_sub_tables[~(all_sub_tables == 0).all(axis=1)] # remove rows (features) with no signal because above the blank threshold
    all_sub_tables = all_sub_tables.reset_index()
    
    # Apply the function to filter msn_adj based on the presence of features in all_sub
    msn_df_deblanked = msn_df[msn_df.apply(match_in_all_sub, all_sub_df=all_sub_tables, axis=1)]
    msn_df_deblanked_path = os.path.join(current_project.featurepath, current_project.name + '_msn_deblanked.csv')
    try:
        msn_df_deblanked.to_csv(msn_df_deblanked_path)
    except OSError as exc:
        logging_config.log_exception(
            logger,
            "Failed to persist deblanked feature list to %s",
            msn_df_deblanked_path,
            project=project_context,
            exception=exc,
        )
        raise

    label_tables_list = current_project.label_tables_list
    labels_df = pd.DataFrame([
        {'sample': sample, 'label': table['label']}
        for table in label_tables_list
        for sample in table['samples']
    ])
    msn_df_deblanked = msn_df_deblanked.merge(labels_df, on='sample', how='left') # Step 2: Merge the labels with the main dataframe
    msn_df_deblanked['label'] = msn_df_deblanked.groupby('sample')['label'].transform(lambda x: '&&'.join(x.dropna().unique())) # Step 3: Combine labels for each sample
    msn_df_deblanked['label'] = msn_df_deblanked['label'].apply(lambda x: '&&'.join(sorted(set(x.split('&&'))))) # Ensure labels are unique and sorted
    msn_df_deblanked = msn_df_deblanked.drop_duplicates().reset_index(drop=True)
    try:
        msn_df_deblanked.to_csv(msn_df_deblanked_path)
    except OSError as exc:
        logging_config.log_exception(
            logger,
            "Failed to update deblanked feature list at %s",
            msn_df_deblanked_path,
            project=project_context,
            exception=exc,
        )
        raise

    if current_project.meta:
        feature_grouping(msn_df_deblanked, Rt_threshold, Correlation_threshold) # important part, where if multiple template have been treated separately with mzmine, it reasign the features with the sample together based on rt and mz threshold.
        deblanking_featuregroup_info = 'Templates feature grouping complete.'
    else:
        deblanking_featuregroup_info = ''
    current_project.msn_df_deblanked  = msn_df_deblanked
    current_project.treatment = treatment_groups
    current_project.complete = True
    cache.set('project_loaded', current_project)
    current_project.save()

    logging_config.log_info(
        logger,
        "Deblanking completed for project %s. Output saved to %s",
        project_name,
        msn_df_deblanked_path,
        project=project_context,
    )
    
@callback(
    [Output({"type": "popup", "index": 8}, 'children')
     ],
    Input('continue-deblanking-button', 'n_clicks'),
    [State('deblanking-level', 'value'),
     State('rt-threshold-templates', 'value'),
     State('corr-threshold-templates', 'value')],
    prevent_initial_call = True
         )
def continue_to_deblanking(n_clicks, Level, rt_threshold, correlation_threshold): # Level is the blank to sample ratio signal, default is 5
    global current_project
    if n_clicks == 1:
        project_name = getattr(current_project, 'name', 'unknown')
        project_context = current_project if getattr(current_project, "update_log", None) else None
        logging_config.log_info(
            logger,
            "Deblanking requested for project %s",
            project_name,
            project=project_context,
        )
        try:
            deblank_and_grouping(Level, rt_threshold, correlation_threshold)
        except FileNotFoundError as exc:
            missing_file = getattr(exc, 'filename', None) or str(exc)
            logging_config.log_exception(
                logger,
                "Deblanking aborted for project %s. Missing feature list: %s",
                project_name,
                missing_file,
                project=project_context,
                exception=exc,
            )
            return dbc.Alert(
                f"The feature list {missing_file} could not be found. Please verify the file exists before continuing.",
                color="danger",
                className="mb-0",
            )
        except Exception as exc:
            logging_config.log_exception(
                logger,
                "Unexpected error while deblanking project %s",
                project_name,
                project=project_context,
                exception=exc,
            )
            return dbc.Alert(
                "An unexpected error occurred during blank signal removal. Check the logs for more information.",
                color="danger",
                className="mb-0",
            )
        logging_config.log_info(
            logger,
            "Deblanking finished for project %s",
            project_name,
            project=project_context,
        )
        return dash.no_update
    else:
        if current_project.complete == True:
            return dash.no_update
    return dash.no_update


button_style = {
    'display': 'flex',  # Use flexbox for layout
    'align-items': 'center',  # Vertically center the flex items
    'justify-content': 'center',  # Horizontally center the flex items
    'width': '35vw',  # Adjust width as needed
    'paddingTop': '0',  # Adjust or remove if using flexbox for centering
    'height': '8vw',  # Adjust based on your layout needs
    'margin': 'auto',
    'font-size': '1.75vw',
    'font-weight': 'bold',
    'border-width': '3px'
}

return_main = html.Div([
                    dbc.Button('Return to main menu for analytics', id='up-return-main-btn', color="warning", href = '/home', style = button_style),
                    html.Br()
                        ])


main_layout = html.Div([
                html.Div(id="output-container"),
                html.Div(id="popup", style={'display': 'none'}),
                html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
                html.Div([
                    html.Div(id='scroll-trigger', style={'display': 'none'}), # Add a hidden div that will be used to trigger the scroll,
                    html.Div(id='scroll-trigger2', style={'display': 'none'}),
                    html.Div(id='scroll-trigger3', style={'display': 'none'}),
                    html.Div(id='scroll-trigger4', style={'display': 'none'}),
                    html.Div(id='scroll-trigger5', style={'display': 'none'}),
                    html.Div(id='scroll-trigger6', style={'display': 'none'}),
                    html.Div(id='scroll-trigger7', style={'display': 'none'}),
                    html.Div(id='scroll-trigger8', style={'display': 'none'}),
                    html.Div(id='scroll-trigger9', style={'display': 'none'}),
                    html.Div(id='scroll-trigger10', style={'display': 'none'}),
                    html.Div(id='scroll-trigger11', style={'display': 'none'}),
                    html.Div(id='scroll-trigger12', style={'display': 'none'}),
                    popup_qa,
                    html.Br(),
                    title,
                    html.Br(),
                    dir_input,
                        ])
                      ], style ={'margin-top': '50px'})

