# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:57:47 2024

@author: Ronan
"""

import dash, os
from diskcache import Cache
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, callback_context, callback

cache = Cache('./disk_cache')

dash.register_page(__name__)

project_loaded = cache.get('project_loaded')



navbar = dbc.Navbar(
    dbc.Container([
            html.A(
            dbc.Row([
                dbc.Col(html.Img(src='/assets/logo.png', height="40px")),
                dbc.Col(
                    dbc.NavbarBrand("", className="ms-2",
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
            "Main menu",
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

main_page = html.Div([
    dbc.Row([
        dbc.Col(dbc.Button('Untargeted MS pipeline', id='un-ms-pip', href = '/untargeted_menu', color="info", style=button_style),
                width=6, className="d-flex justify-content-center align-items-center"),
        dbc.Col(dbc.Button('Learn', href = '/learn', outline=True, color="secondary", style=button_style),
                width=6, className="d-flex justify-content-center align-items-center"),
    ], className="my-3", justify="center"),
    dbc.Row(
        dbc.Col(html.H3('Main menu', style={'textAlign': 'center'}),
                className="d-flex justify-content-center align-items-center",
                style={'height': '50px'}),  # Adjust the height as needed
        className="w-100"
    ),
    dbc.Row([
        dbc.Col(dbc.Button("Analytics", id = 'analytics', href = '/analytics', outline=True, color="success", style=button_style),
                width=6, className="d-flex justify-content-center align-items-center"),
        dbc.Col(dbc.Button("Exit MassLearn", href = '/login', outline=True, color="warning", style=button_style),
                width=6, className="d-flex justify-content-center align-items-center"),
    ], className="my-3", justify="center")
])

# Callback to adapt the button Analytics outline (visibility) when a project is loaded
@callback(
    [Output("analytics", "outline"),
     Output("un-ms-pip", "outline")],
    Input("url", "pathname"))
def change_outline(pathname):
    project_loaded = cache.get('project_loaded')
    if project_loaded != None:
        if project_loaded.complete:
            return False, True
        else:
            return True, False
    else:
        return True, False
    


navbar_height = '50px'
def get_layout():
    if cache.get('identity') is not None:
        return html.Div([
            html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
            html.Div([
                html.Div(main_page, style={'margin-top': navbar_height}),  # Adjust content start below navbar
            ], style={
                'position': 'relative',  # Ensure content is positioned relative to the navbar
                'top': navbar_height,  # Start content below the navbar
                'overflow': 'auto',  # Allow content to be scrollable if it exceeds the viewport height
                'max-height': f'calc(100vh - {navbar_height})'  # Max height to prevent overflow
            }, className="container-fluid")  # Use container-fluid for full width
        ])
    else:
        return dcc.Link('Go to login', href='/login')
        

layout = dcc.Link('Go to login', href='/login')

