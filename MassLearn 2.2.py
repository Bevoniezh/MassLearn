# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:01:51 2024

@author: Ronan
"""

import os
import re
import lz4
import dash
import shutil
import pickle
from diskcache import Cache
import Modules.cache_manager as cm
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback_context, callback
from dash.dependencies import Input, Output, State, ClientsideFunction

masslearn_logo = "/assets/logo.png"
Log = cm.LogManager()
cache_path = './disk_cache'
current_path= os.getcwd()
import Modules.file_manager
Modules.file_manager.MassLearn_directory = current_path

try:
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)  # Remove the entire cache directory
        os.makedirs(cache_path)  # Recreate the cache directory if needed
except PermissionError as e:
    pass

cache = Cache(cache_path) # this is the disk cache for session variable in the dash app
cache.clear()
# Clear the cache
cache.set('logo', masslearn_logo)
cache.set('log', Log)
cache.set('identity', None)
cache.set('project', {})
print(f'project cache app: {cache.get("project")}')
print(f"current path: {current_path}")

# remove this
# with lz4.frame.open(r"C:\Users\Ronam\Desktop\Working_folder\Bioinformatic_projects\Molecular Network project\Projects\mock\project.masslearn", "rb") as file:
#       project_loaded = pickle.load(file)
# cache.set('project_loaded', project_loaded)
cache.set('project_loaded', None )
cache.set('raw_dir_path', None)
cache.set('switch_QA_status', False)

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.QUARTZ], suppress_callback_exceptions=True)

# Client function to scroll automatically down when a button is triggered
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger', 'children'),
    [Input('project-name-input', 'valid')]
)     
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger2', 'children'),
    [Input('raw-dir-input', 'valid'),
     Input('mzml-manual-input', 'valid')]
)     
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger3', 'children'),
    Input("convert-range-button", "n_clicks")
)
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger4', 'children'),
    Input("noise-trace-button", "n_clicks")
)    
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger5', 'children'),
   Input("ms-noise-button", "n_clicks")     
)
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger6', 'children'),
    Input("up-mzmine-button", "n_clicks")
)
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger7', 'children'),
    Input("mzml-alternative", "n_clicks")
)
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger8', 'children'),
    Input("generate-button", "n_clicks")
)    
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger9', 'children'),
    Input("continue-template-button", "n_clicks")
) 
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger10', 'children'),
    Input("meta-nomore-template-button", "n_clicks")
) 
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger11', 'children'),
    Input("skip-button", "n_clicks")
) 
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    Output('scroll-trigger12', 'children'),
    Input("return-main-part", "children")
)
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='scrollToBottom'),
    [Output('scroll-trigger-login', 'children')],
    [Input("seems-button", "n_clicks"),
    Input("mzmine-button", "n_clicks"),
    Input("msconvert-button", "n_clicks")]
)


# Pages and cache definition
from pages import login, home, analytics, untargeted_menu, untargeted_pipeline, learn


#cache.set('software', softwares)          


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),    
    html.Div(id='page-content'),
                            ])
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
)
def display_page(pathname):
    if pathname == '/home':
        return home.get_layout()
    if pathname == '/login':
        return login.layout
    elif pathname == '/analytics':
        return analytics.get_layout()
    elif pathname == '/untargeted_menu':
        return untargeted_menu.get_layout()
    elif pathname == '/untargeted_pipeline':
        return untargeted_pipeline.get_layout()
    elif pathname == '/learn':
        return learn.get_layout()
    else:
        return '404'
    
if __name__ == '__main__':
    app.run(debug=False)         

