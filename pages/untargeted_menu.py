# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:43:09 2024

@author: Ronan
"""
import re
import os
import lz4
import dash
import pickle
from diskcache import Cache
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, callback_context, callback

cache = Cache('./disk_cache')

dash.register_page(__name__)

# Definition of navbar
###############################################################################
project_loaded = cache.get('project_loaded')


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
            "Untargeted MS pipeline menu",
            width="auto",
            className="d-flex justify-content-center",
            style={"fontSize": "20px", "fontWeight": "bold"}  # bigger font
        ),
    ]),
    color="dark",
    dark=True,
    style={'height': '50px'},
)


button_style = {
    'display': 'flex',  # Use flexbox for layout
    'align-items': 'center',  # Vertically center the flex items
    'justify-content': 'center',  # Horizontally center the flex items
    'width': '15vw',  # Adjust width as needed
    'paddingTop': '0',  # Adjust or remove if using flexbox for centering
    'height': '10vw',  # Adjust based on your layout needs
    'margin': 'auto',
    'font-size': '1.25vw',
    'font-weight': 'bold',
    'border-width': '3px'
}

project_non_loaded = dbc.Row([
                        dbc.Col(dbc.Button('Create a new project', href = '/untargeted_pipeline', color="info", style=button_style),
                                width=6, className="d-flex justify-content-center align-items-center"),
                        dbc.Col(dbc.Button('Load a project', id='load', outline=True, color="primary", style=button_style),
                                width=6, className="d-flex justify-content-center align-items-center"),
                    ], className="my-3", justify="center")

title = dbc.Row(
    dbc.Col(html.H3('Untargeted MS pipeline menu', style={'textAlign': 'center'}),
            className="d-flex justify-content-center align-items-center",
            style={'height': '50px'}),  # Adjust the height as needed
    className="w-100"
)
    
exit_options = dbc.Row([
                dbc.Col(dbc.Button('Return to Main Menu', href = '/home', outline=True, color="warning", style=button_style),
                        width=6, className="d-flex justify-content-center align-items-center"),
                dbc.Col(dbc.Button("Exit MassLearn", href = '/login', outline=True, color="warning", style=button_style),
                        width=6, className="d-flex justify-content-center align-items-center"),
            ], className="my-3", justify="center")

modal_win = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Load a project")),
                html.Div([
                    html.H5('Paste or write the path to the project.masslearn file you want to load and PRESS ENTER', style={'textAlign': 'center'}),
                    dbc.Input(id="load-input", valid = None, placeholder=r"C:\Users\Arthur\project1.massLearn", type="text", style={'maxWidth': '600px'}),
                    html.Br(),
                    dbc.Spinner(html.P('No files loaded', id="load-output"))
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',      # Center horizontally in the flex container
                    }),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="modal",
            is_open=False,
        ),
    ])

def normalize_path(input_path = ''):
    # Check if the path is enclosed in quotation marks and remove them
    if input_path == None:
        input_path = ''
    if input_path.startswith(("'", '"')) and input_path.endswith(("'", '"')):
        # Remove the first and last characters (the quotation marks)
        return input_path[1:-1]
    return input_path

@callback(
    Output("modal", "is_open"),
    [Input("load", "n_clicks"), 
     Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    [Output("load-output", "children"),
    Output("load-input", "valid"),
    Output("load-input", "invalid"),
    Output("load-input", "disabled")],
    [Input("load-input", "n_submit")],
    [State("load-input", "value")],
    prevent_initial_call = True
)                  
def validate_project(n_submit, file):
    file = normalize_path(file) # to handle the potential "" 
    if n_submit:
       # Check if the file path ends with ".masslearn"
        if file.endswith('.masslearn'):
            # Check if the file exists
            if os.path.exists(file):
                # Try to open the file as a pickle binary file
                try:
                    with lz4.frame.open(file, 'rb') as f:
                        # Try to load the pickle object
                        project_file = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    return "Invalid file.", None, True, None
                if project_file.treatment:
                    cache.set('project_loaded', project_file) 
                    print(f"um: {cache.get('project_loaded')}")                          
                    return "Valid file, you can close the pop-up", True, None, True
                # If successful, return True
                else:                            
                    return "The untargeted pipeline from this file is not finished, you must repeat the pipeline.", None, True, None
    return "", None, None, None


# Callback to add new project as main project
@callback(
    Output("um-project-menu", "children"),
    [Input("load-input", "valid"), # if a project is loaded in the project loading module
     Input('url', 'pathname')],
     State("load-input", "value"),
    prevent_initial_call = True
)
def add_dropdown_item(valid, reload, project_path):  
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "load-input":
        if valid == None:    
            project_loaded = cache.get('project_loaded')
            if project_loaded != None:
                label = project_loaded.name
            else:
                label = 'Create or load a project'
            return label
        elif valid != None:
            label = os.path.splitext(os.path.basename(project_path))[0]
            return label  
    else:
        project_loaded = cache.get('project_loaded')
        if project_loaded != None:
            label = project_loaded.name
        else:
            label = 'Create or load a project'
        return label
    


def get_layout():
    navbar_height = "50px"  # Adjust based on your actual navbar height
    if cache.get('identity') is not None:
        content = project_non_loaded
        um_layout = html.Div([
            modal_win,
            dcc.Store(id='um-qa-switch-store', storage_type='session'),
            html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
            html.Div([
                html.Div(content, style={'margin-top': navbar_height}),  # Adjust content start below navbar
                title,
                exit_options,
            ], style={
                'position': 'relative',  # Ensure content is positioned relative to the navbar
                'top': navbar_height,  # Start content below the navbar
                'overflow': 'auto',  # Allow content to be scrollable if it exceeds the viewport height
                'max-height': f'calc(100vh - {navbar_height})'  # Max height to prevent overflow
            }, className="container-fluid"),  # Use container-fluid for full width
            
        ])
        return um_layout
    else:
        return dcc.Link('Go to login', href='/login')
         

layout = get_layout()







