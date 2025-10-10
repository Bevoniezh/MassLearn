# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:57:47 2024

@author: Ronan
"""

import dash
from diskcache import Cache
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc, dash_table, callback_context, callback
import pandas as pd 
import Modules.questions as Q
import random
from dash.exceptions import PreventUpdate

cache = Cache('./disk_cache')

dash.register_page(__name__)

navbar = dbc.Navbar(
    dbc.Container([
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row([
                        dbc.Col(html.Img(src='/assets/logo.png', height="40px")),
                        dbc.Col(dbc.NavbarBrand("<- -        Learn", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/home",
                style={"textDecoration": "none"},
            ),
            
            dbc.Switch(
                    label="Activate Q&A",
                   value=True,
                   disabled = True),
                ]),
    color="dark",
    dark=True,
    style={'height': '50px'},)




# Definition of pop up Q&A
###############################################################################

popup_qa = html.Div([
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(id = 'learn-modal_title')),
                dbc.ModalBody(id = 'learn-modal_body'),
                dbc.ModalFooter([
                    html.Div([
                    dbc.Button(id="learn-answer_a", color="secondary", n_clicks=0),
                    dbc.Button(id="learn-answer_b", color="secondary", n_clicks=0),
                    dbc.Button(id="learn-answer_c", color="secondary", n_clicks=0),
                                ], className="d-grid gap-2"),
                    html.Div([dbc.Button(children = "Next question", id="next", color="primary", n_clicks=0)]),
                    html.Div(children = 'Score : 0/0', id = 'score'),
                    html.Div(children = 'Nb of errors : 0', id = 'errors')
                    ], style={
                            'display': 'flex',
                            'flexDirection': 'column',
                            'alignItems': 'center',      # Center horizontally in the flex container
                                    })
                    ], id="learn-modal", is_open=False)
                ])

answer_save = pd.DataFrame()
correct_answer = ''
answer_1 = ''
answer_2 = ''
answer_3 = ''
successes = 0
nb_questions = 0
nb_errors = 0
questions_answered = {} # questions as keys, success as values

# TODO: delay automtically the closing of the modal.  (not is_open) close it
@callback(
    [Output("learn-modal", "is_open"),
     Output("learn-modal_title", "children"),
     Output("learn-modal_body", "children"),
     Output("learn-answer_a", "children"),
     Output("learn-answer_b", "children"),
     Output("learn-answer_c", "children"),
     Output("learn-answer_a", "color"),
     Output("learn-answer_b", "color"),
     Output("learn-answer_c", "color"),
     Output("learn-answer_a", "disabled"),
     Output("learn-answer_b", "disabled"),
     Output("learn-answer_c", "disabled"),
     Output("learn-answer_a", "n_clicks"),
     Output("learn-answer_b", "n_clicks"),
     Output("learn-answer_c", "n_clicks"),
     Output("score", "children"),
     Output("errors", "children")],
    [Input("play", "n_clicks"),
     Input("next", "n_clicks"),
     Input("learn-answer_a", "n_clicks"),
     Input("learn-answer_b", "n_clicks"),
     Input("learn-answer_c", "n_clicks")],    
    [State("learn-modal", "is_open"),],
    prevent_initial_call = True
)
def open_modal(play_click, next_click, answer_a, answer_b, answer_c, is_open):
    ctx = dash.callback_context    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    global answer_save, answer_1, answer_2, answer_3, successes, nb_questions, questions_answered, correct_answer, nb_errors
    if button_id == 'play' and not(is_open):
        successes = 0
        nb_questions = 0
        questions_answered = {} # question answered is here to prevent repeating to display the same question in a session
        nb_errors = 0
        
    q = Q.Questions()
    new_question = False
    if answer_save.empty:
        while new_question == False:            
            question_row = q.random_row
            answer_save = question_row
            title = answer_save['Theme'].iloc[0]
            if title in list(questions_answered.keys()):
                if questions_answered[title]: # if there is already a valid answer because the value should be True
                    new_question = False
                    continue
                else:
                    new_question = True
                    continue
            else:
                new_question = True
        
        correct = answer_save['Correct Answer'].iloc[0] - 3 # we remove 3 because of the number of column in th question df
        answers = [answer_save['Answer 1'].iloc[0], answer_save['Answer 2'].iloc[0], answer_save['Answer 3'].iloc[0]]
        correct_answer = answers[correct] 
        
        answer_1 = random.choice(answers)
        answer_1_color = "secondary"
        answers.remove(answer_1) # remove the chosen answer
        
        answer_2 = random.choice(answers)
        answer_2_color = "secondary"
        answers.remove(answer_2)
        
        answer_3 = answers[0]
        answer_3_color = "secondary"

    
    question = answer_save['Question'].iloc[0]
    title = answer_save['Theme'].iloc[0]
    success = True
    if button_id != 'next' and button_id != 'play':        
        if answer_1 == correct_answer:
            answer_1_color = "success"
        elif answer_1 != correct_answer and answer_a == 1: 
            answer_1_color = "danger"
            success = False
        else:
            answer_1_color = "secondary"
        
        if answer_2 == correct_answer:
            answer_2_color = "success"
        elif answer_2 != correct_answer and answer_b == 1: 
            answer_2_color = "danger"
            success = False
        else: 
            answer_2_color = "secondary"
        
        if answer_3 == correct_answer:
            answer_3_color = "success"
        elif answer_3 != correct_answer and answer_c == 1: 
            answer_3_color = "danger"
            success = False
        else: 
            answer_3_color = "secondary"

        if success:
            questions_answered[question] = True # if there is a success
        else: 
            questions_answered[question] = False
            nb_errors += 1
            
        successes = sum(questions_answered.values()) # how many valid answer
        nb_questions = len(questions_answered.keys())
        answer_save = pd.DataFrame()
        
        return True, title, question, answer_1, answer_2, answer_3, answer_1_color, answer_2_color, answer_3_color, True, True, True, 0, 0, 0, f'Score : {successes} / {nb_questions}', f'Nb of errors : {nb_errors}'
    else:      
        return True, title, question, answer_1, answer_2, answer_3, "secondary", "secondary", "secondary", None, None, None, 0, 0, 0, f'Score : {successes} / {nb_questions}', f'Nb of errors : {nb_errors}' # return invert of the current state of the modal, and the modal parameters




navbar_height = '50px'
def get_layout():
    if cache.get('identity') is not None:
        return html.Div([
            popup_qa,
            html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
            html.Div([dbc.Button("I want to learn!", id='play', n_clicks=0, color="primary"),
                                  ], 
                    style={'position': 'absolute',  # Positioning the div absolutely inside its relative parent
                            'top': '50%',            # Centering vertically
                            'left': '50%',           # Centering horizontally
                            'transform': 'translate(-50%, -50%)'  # Adjusting the exact center position
                        }),
        ])
    else:
        return dcc.Link('Go to login', href='/login')
        

layout = dcc.Link('Go to login', href='/login')

