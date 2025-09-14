# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:01:51 2024

@author: Ronan
"""


import os
import dash
import tkinter as tk
from diskcache import Cache
from tkinter import filedialog
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, callback_context, callback


cache = Cache('./disk_cache')

dash.register_page(__name__)

with open('./Cache/user.dat') as fic: # import userlist from user.dat
    userlist = fic.read().split('\n')
    

userlist = sorted(userlist)
dropdown_items = [{"label": text, "value": text} for text in userlist]
select = html.Div([
            dbc.Select(
                id="select",
                options=dropdown_items,
                style={'maxWidth': '400px'}
            ),
            html.Br(),
            html.Div(id="enter-button")
            ], style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',      # Center horizontally in the flex container
            })



def check_software():
    software_list = [cache.get('seems.exe'), cache.get('MZmine.exe'), cache.get('msconvert.exe')]
    message = []
    confirmation = True
    for s, ref in zip(software_list, ['seems.exe', 'MZmine.exe', 'msconvert.exe']):
        if s == None:
            message.append(f'{ref} is NOT defined. Upload this file from the software main folder.')
            confirmation == False
        else:
            message.append(f'{ref} is defined. If the software fails to load, you might define again this file.')
    return message, confirmation
            
# Define the modal for loading software path
###############################################################################


text_input = html.Div([
    html.Div([
        dbc.Input(id="input", valid = None, placeholder="Firstname Lastname", type="text", style={'maxWidth': '400px'}),
        html.Br(),
        html.P(id="validation-output")
        ], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',      # Center horizontally in the flex container
        }),
        ])
            
software = html.Div([   
            html.H5('Powered by', style={'textAlign': 'center'}),      
            html.Div([
                html.Button(
                    children=[html.Img(src='/assets/seems.png', style={'width': 'auto', 'height': '50px'})],
                    id='seems-button',
                    style={'background': 'none', 'border': 'none', 'padding': '0'},
                    n_clicks=0
                ),
                dbc.Tooltip(
                    check_software()[0][0],
                    target="seems-button",  # ID of the component to which the tooltip is attached
                    placement="left"),
                html.Button(
                    children=[html.Img(src='/assets/mzmine_logo.png', style={'width': 'auto', 'height': '50px'})],
                    id='mzmine-button',
                    style={'background': 'none', 'border': 'none', 'padding': '0'},
                    n_clicks=0
                ),
               dbc.Tooltip(
                   check_software()[0][1],
                   target="mzmine-button",  # ID of the component to which the tooltip is attached
                   placement="left"),
               html.Button(
                   children=[html.Img(src='/assets/proteowizard.jpg', style={'width': 'auto', 'height': '50px'})],
                   id='msconvert-button',
                   style={'background': 'none', 'border': 'none', 'padding': '0'},
                   n_clicks=0
               ),
                dbc.Tooltip(
                    check_software()[0][2],
                    target="msconvert-button",  # ID of the component to which the tooltip is attached
                    placement="left"),
                ], style={
                    'display': 'flex', 'justify-content': 'space-around', 'width': '100%',
                }),
            html.Br(),        
            ])
                    
input_temp = html.Div(children = [],
                id = 'input-div', style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',      # Center horizontally in the flex container
                })

                    
layout = html.Div([
                html.H1(''),
                html.H1('Welcome to MassLearn', style={'textAlign': 'center'}),
                html.H3('Select a user', style={'textAlign': 'center'}),
                select,
                html.H1(''),
                html.H3('or', style={'textAlign': 'center'}),
                html.H3('add a new user', style={'textAlign': 'center'}),
                text_input,
                software,   
                input_temp,
                html.Div(id='scroll-trigger-login', style={'display': 'none'}), # Add a hidden div that will be used to trigger the scroll
                ])

cache.set('software_path', None) # cache to reemeber which software path is to set by the user
@callback(
    [Output("input-div", "children")],
    [Input("seems-button", "n_clicks"),
     Input("mzmine-button", "n_clicks"),
     Input("msconvert-button", "n_clicks")]
)
def open_input_software(n_clicks_seems, n_clicks_mzmine, n_clicks_msconvert):    
    if n_clicks_seems == 1:
        cache.set('software_path', 'seems.exe')
        lay = [html.Div([html.H5('Indicate the path to the seems.exe file', style={'textAlign': 'center'}),
                dbc.InputGroup([dbc.Input(id="input-software", valid = None, placeholder="path/to/seems.exe", type="text", style={'maxWidth': '600px'}),
                dbc.Button('Confirm file', id='confirm-button', color="primary", n_clicks=0)]),
                html.Br()])]
        return lay
    elif n_clicks_mzmine == 1:
        cache.set('software_path', 'MZmine.exe')
        lay = [html.Div([html.H5('Indicate the path to the MZmine.exe file', style={'textAlign': 'center'}),
                dbc.InputGroup([dbc.Input(id="input-software", valid = None, placeholder="path/to/MZmine.exe", type="text", style={'maxWidth': '600px'}),
                dbc.Button('Confirm file', id='confirm-button', color="primary", n_clicks=0)]),
                html.Br()])]
        return lay
    elif n_clicks_msconvert == 1:
        cache.set('software_path', 'msconvert.exe')
        lay = [html.Div([html.H5('Indicate the path to the msconvert.exe file', style={'textAlign': 'center'}),
                dbc.InputGroup([dbc.Input(id="input-software", valid = None, placeholder="path/to/msconvert.exe", type="text", style={'maxWidth': '600px'}),
                dbc.Button('Confirm file', id='confirm-button', color="primary", n_clicks=0)]),
                html.Br()])]
        return lay
    else:
        return ['']


@callback(
    [Output("seems-button", "n_clicks"),
     Output("mzmine-button", "n_clicks"),
     Output("msconvert-button", "n_clicks")],
    Input("confirm-button", "n_clicks"),
    State("input-software", "value"),   
    prevent_initial_call=True
)
def software_output(confirm, input_soft): 
    if confirm != 0:
        if input_soft != None:
            softwares = cache.get('software')
            soft = cache.get('software_path')
            if soft == 'seems.exe':
                newline = softwares[0].split(' # ')[0] + ' # ' + input_soft
                softwares[0] = newline
            elif soft == 'MZmine.exe':
                newline = softwares[1].split(' # ')[0] + ' # ' + input_soft
                softwares[1] = newline
            elif soft == 'msconvert.exe':
                newline = softwares[2].split(' # ')[0] + ' # ' + input_soft
                softwares[2] = newline
            with open('Cache/software_path_dash.dat', 'w') as fic:
                fic.write(f"{softwares[0]}\n")
                fic.write(f"{softwares[1]}\n")
                fic.write(f"{softwares[2]}")
        return 0, 0, 0
    else:
        return 0, 0, 0
    


@callback(
    [Output("validation-output", "children"),
    Output("input", "valid"),
    Output("input", "invalid"),
    Output("select", "options")],
    [Input("input", "n_submit")],
    [State("input", "value")]
)
def validate_input(n_submit, value):
    global userlist
    global dropdown_items
    if n_submit:
        # Add your validation logic here
        if  (len(value) > 6) and (' ' in value) and (value not in userlist):
            userlist.append(value)
            userlist = sorted(userlist)
            dropdown_items = [{"label": text, "value": text} for text in userlist]
            
            with open('./Cache/user.dat', 'w') as fic: # import userlist from user.dat
                userlist_towrite = '\n'.join(userlist)
                fic.write(userlist_towrite)
            
            return f"{value} successfully added! Now go back in the selection list", True, None, dropdown_items
        elif value in userlist:
            f"{value} already exist. Select it in the name list for connecting to MassLearn.", None, True, dropdown_items
        else:
            return "Total lenght of your name should be at least 6 letter (first and last name), separated with a space. Ex: arthur smith", None, True, dropdown_items
    return "", None, None, dropdown_items


@callback(
    Output('enter-button', 'children'),
    Input("select", "value")
   )
def validate_selection(value):
    if value is not None:  
        if check_software()[1] == True:
            log = f'Selected user: {value}'
            Log = cache.get('log')
            cache.set('identity', value)
            Log.user = value
            Log.update(log)
            cache.set('log', Log)
            return dbc.Button("Enter MassLearn", color="primary", href = '/home')
        else:
            return dbc.ListGroupItem("Make sure all the software path below are defined, and try again.", color="warning", style={'maxWidth': '600px', 'fontSize': '12px', 'padding-left': '5px','padding-right': '5px'})
    else:
        return ''


