# -*- coding: utf-8 -*-
"""
MassLearn Login Page
@author: Ronan
"""

import os
import json
import configparser
import dash
import threading, time, os
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from diskcache import Cache

# -------------------------------------------------------------------
# Directories
BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CONFIG_FILE = os.path.join(CONFIG_DIR, "config.ini")
USERS_FILE = os.path.join(CONFIG_DIR, "users.json")

# -------------------------------------------------------------------
# Initialize config.ini if missing
config = configparser.ConfigParser()

if not os.path.exists(CONFIG_FILE):
    config["Software"] = {
        "SeeMS": "",
        "MZmine": "",
        "ProteoWizard": ""
    }
    config["General"] = {
        "last_user": ""
    }
    with open(CONFIG_FILE, "w") as f:
        config.write(f)

config.read(CONFIG_FILE)

# -------------------------------------------------------------------
# Initialize users.json if missing
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": []}, f, indent=2)

with open(USERS_FILE) as f:
    userlist = json.load(f)["users"]

# -------------------------------------------------------------------
# DiskCache for runtime
cache = Cache("./disk_cache")
dash.register_page(__name__)

# -------------------------------------------------------------------
# Helpers
def save_users(userlist):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": sorted(userlist)}, f, indent=2)

def update_config(section, key, value):
    config.set(section, key, value)
    with open(CONFIG_FILE, "w") as f:
        config.write(f)

def check_software():
    """Check if all software paths are set and exist."""
    message = []
    confirmation = True

    for key, label in [
        ("SeeMS", "seems"),
        ("MZmine", "mzmine"),
        ("ProteoWizard", "msconvert")
    ]:
        raw_path = config.get("Software", key, fallback="").strip()

        # Remove quotes if present
        path = raw_path.strip('"').strip("'")

        # Normalize path to absolute
        path = os.path.abspath(os.path.normpath(path))
        print(path)
        if not path or not os.path.exists(path):
            message.append(f"{label} is NOT defined or invalid: {raw_path}")
            confirmation = False
        else:
            message.append(f"{label} is defined at {path}.")

    return message, confirmation


# -------------------------------------------------------------------
# UI elements

dropdown_items = [{"label": text, "value": text} for text in sorted(userlist)]

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
    'alignItems': 'center',
})

text_input = html.Div([
    dbc.Input(
        id="input",
        valid=None,
        placeholder="Firstname Lastname",
        type="text",
        style={'maxWidth': '400px'}
    ),
    html.Br(),
    html.P(id="validation-output")
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
})

software = html.Div([
    html.H5('Powered by', style={'textAlign': 'center'}),
    html.Div([
        html.Button(
            children=[html.Img(src='/assets/seems.png', style={'height': '50px'})],
            id='seems-button',
            style={'background': 'none', 'border': 'none', 'padding': '0'},
            n_clicks=0
        ),
        dbc.Tooltip(check_software()[0][0], target="seems-button", placement="left"),

        html.Button(
            children=[html.Img(src='/assets/mzmine_logo.png', style={'height': '50px'})],
            id='mzmine-button',
            style={'background': 'none', 'border': 'none', 'padding': '0'},
            n_clicks=0
        ),
        dbc.Tooltip(check_software()[0][1], target="mzmine-button", placement="left"),

        html.Button(
            children=[html.Img(src='/assets/proteowizard.jpg', style={'height': '50px'})],
            id='msconvert-button',
            style={'background': 'none', 'border': 'none', 'padding': '0'},
            n_clicks=0
        ),
        dbc.Tooltip(check_software()[0][2], target="msconvert-button", placement="left"),
    ], style={'display': 'flex', 'justify-content': 'space-around', 'width': '100%'}),
    html.Br(),
])

shutdown_button = html.Div([
    html.Br(),
    dbc.Button("Shutdown MassLearn", id="shutdown-button", color="danger")
], style={'textAlign': 'center', 'marginTop': '20px'})

input_temp = html.Div(id='input-div', style={
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
})

layout = html.Div([
    html.H1('Welcome to MassLearn', style={'textAlign': 'center'}),
    html.H3('Select a user', style={'textAlign': 'center'}),
    select,
    html.H3('or', style={'textAlign': 'center'}),
    html.H3('add a new user', style={'textAlign': 'center'}),
    text_input,
    software,
    html.Div(id="shutdown-output", style={'textAlign': 'center'}),  # add this line
    html.Br(),
    html.Div(
        dbc.Button(
            "Shutdown MassLearn",
            id="shutdown-button",
            color="danger",
            style={'minWidth': '200px'}
        ),
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'marginTop': '10px'
        }
    ),input_temp,
    html.Div(id='scroll-trigger-login', style={'display': 'none'}),
])

# -------------------------------------------------------------------
# Callbacks

@callback(
    Output("input-div", "children"),
    [Input("seems-button", "n_clicks"),
     Input("mzmine-button", "n_clicks"),
     Input("msconvert-button", "n_clicks")]
)
def open_input_software(n_clicks_seems, n_clicks_mzmine, n_clicks_msconvert):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ['']

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "seems-button":
        cache.set('software_path', "SeeMS")
        label = "seems.exe"
    elif button_id == "mzmine-button":
        cache.set('software_path', "MZmine")
        label = "MZmine.exe"
    elif button_id == "msconvert-button":
        cache.set('software_path', "ProteoWizard")
        label = "msconvert.exe"
    else:
        return ['']

    return [html.Div([
        html.H5(f'Indicate the path to {label}', style={'textAlign': 'center'}),
        dbc.InputGroup([
            dbc.Input(id="input-software", placeholder=f"path/to/{label}", type="text", style={'maxWidth': '600px'}),
            dbc.Button('Confirm file', id='confirm-button', color="primary", n_clicks=0)
        ]),
        html.Br()
    ])]

@callback(
    [Output("shutdown-output", "children"),
     Output("shutdown-button", "style")],
    Input("shutdown-button", "n_clicks"),
    prevent_initial_call=True
)
def shutdown_app(n):
    if n:
        def stop_server():
            time.sleep(1)
            os._exit(0)

        threading.Thread(target=stop_server).start()

        return (html.Div([
            html.H2("MassLearn process is down, please close the window...", 
                    style={'textAlign': 'center', 'marginTop': '40px'}),
            html.Script("setTimeout(()=>{document.body.innerHTML=''; window.close();}, 1500);")
        ]), {'display': 'none'})
    return "", {'display': 'inline-block'}




@callback(
    [Output("seems-button", "n_clicks"),
     Output("mzmine-button", "n_clicks"),
     Output("msconvert-button", "n_clicks")],
    Input("confirm-button", "n_clicks"),
    State("input-software", "value"),
    prevent_initial_call=True
)
def software_output(confirm, input_soft):
    if confirm and input_soft:
        soft = cache.get("software_path")
        update_config("Software", soft, input_soft)
    return 0, 0, 0

@callback(
    [Output("validation-output", "children"),
     Output("input", "valid"),
     Output("input", "invalid"),
     Output("select", "options")],
    Input("input", "n_submit"),
    State("input", "value")
)
def validate_input(n_submit, value):
    global userlist, dropdown_items
    if n_submit:
        if len(value) > 6 and (' ' in value) and (value not in userlist):
            userlist.append(value)
            save_users(userlist)
            dropdown_items = [{"label": text, "value": text} for text in sorted(userlist)]
            return f"{value} successfully added! Now select it from the list.", True, None, dropdown_items
        elif value in userlist:
            return f"{value} already exists. Select it in the list.", None, True, dropdown_items
        else:
            return "Your name must be at least 6 letters and contain a space. Ex: Arthur Smith", None, True, dropdown_items
    return "", None, None, dropdown_items

@callback(
    Output('enter-button', 'children'),
    Input("select", "value")
)
def validate_selection(value):
    if value:
        message, ok = check_software()
        if ok:
            update_config("General", "last_user", value)
            log = f'Selected user: {value}'
            Log = cache.get('log')
            cache.set('identity', value)
            cache.set('current_learn_session', None)
            Log.user = value
            Log.update(log)
            cache.set('log', Log)
            return dbc.Button("Enter MassLearn", color="primary", href='/home')
        else:
            return dbc.ListGroupItem(f"Please define missing software path by cliking on the icons below, then restart MassLearn.\n\nThe missing softwares are:{' - '.join(message)}", 
                                     color="warning", style={'maxWidth': '600px', 'fontSize': '12px'})
    return ""
