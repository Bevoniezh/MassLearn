# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:57:47 2024

@author: Ronan
"""

import json
import random
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, dash_table, html, callback
from dash.dependencies import Input, Output, State
from diskcache import Cache

import Modules.questions as Q

cache = Cache('./disk_cache')
SESSION_CACHE_KEY = 'current_learn_session'
SESSION_USER_KEY = 'current_learn_user'
BASE_DIR = Path(__file__).resolve().parent.parent
USERS_FILE = BASE_DIR / 'users.json'

current_session_id = cache.get(SESSION_CACHE_KEY)


def _normalise_sessions(raw_sessions):
    """Return a list of session dictionaries regardless of the legacy format."""
    if isinstance(raw_sessions, list):
        return [session for session in raw_sessions if isinstance(session, dict)]
    if isinstance(raw_sessions, dict):
        sessions = [session for session in raw_sessions.values() if isinstance(session, dict)]
        sessions.sort(key=lambda item: item.get('started_at', '') or item.get('session_id', ''))
        return sessions
    return []


def _load_users_file():
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open('r', encoding='utf-8') as users_file:
                data = json.load(users_file)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    if isinstance(data, dict) and 'users' in data:
        if isinstance(data['users'], dict):
            data = data['users']
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    normalised = {}
    needs_save = False
    for username, entry in data.items():
        if not isinstance(username, str):
            continue
        if isinstance(entry, dict):
            sessions = _normalise_sessions(entry.get('sessions', []))
            if sessions != entry.get('sessions'):
                needs_save = True
            normalised[username] = {'sessions': sessions}
        elif isinstance(entry, list):
            normalised[username] = {'sessions': _normalise_sessions(entry)}
            needs_save = True
    if needs_save or normalised != data or not USERS_FILE.exists():
        _save_users_file(normalised)
    return normalised


def _save_users_file(data):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open('w', encoding='utf-8') as users_file:
        json.dump(data, users_file, indent=2, sort_keys=True)


def _get_user_entry(data, user):
    entry = data.setdefault(user, {})
    sessions = entry.get('sessions', [])
    normalised = _normalise_sessions(sessions)
    if normalised != sessions:
        entry['sessions'] = normalised
    else:
        entry.setdefault('sessions', [])
    return entry

def _format_timestamp(timestamp):
    if not timestamp:
        return ''
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return timestamp


def _get_session_rows(user):
    rows = []
    if user is None:
        return rows
    data = _load_users_file()
    user_entry = data.get(user, {})
    for session in user_entry.get('sessions', []):
        started_at = _format_timestamp(session.get('started_at'))
        score = session.get('score', {})
        correct = score.get('correct', 0)
        total = score.get('total', 0)
        rows.append({
            'started_at': started_at,
            'score': f"{correct}/{total}"
        })
    rows.sort(key=lambda item: item.get('started_at', ''), reverse=True)
    return rows


def _start_session(user):
    if user is None:
        return None
    data = _load_users_file()
    session_id = datetime.utcnow().isoformat()
    entry = {
        'session_id': session_id,
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'score': {'correct': 0, 'total': 0}
    }

    # ensure correct structure
    user_entry = data.setdefault(user, {'sessions': []})
    sessions = user_entry.setdefault('sessions', [])
    sessions.append(entry)

    _save_users_file(data)
    return session_id



def _update_session_score(user, session_id, correct, total):
    if user is None or session_id is None:
        return
    data = _load_users_file()
    user_entry = data.get(user, {})
    sessions = user_entry.get('sessions', [])

    for session in sessions:
        if session.get('session_id') == session_id:
            session.setdefault('score', {})
            session['score']['correct'] = int(correct)
            session['score']['total'] = int(total)
            session['updated_at'] = datetime.utcnow().isoformat()
            _save_users_file(data)
            return

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
            "Learn",
            width="auto",
            className="d-flex justify-content-center",
            style={"fontSize": "20px", "fontWeight": "bold"}  # bigger font
        ),
    ]),
    color="dark",
    dark=True,
    style={'height': '50px'},
)


# Definition of pop up Q&A
###############################################################################

popup_qa = html.Div([
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(id='learn-modal_title')),
                dbc.ModalBody(id='learn-modal_body'),
                dbc.ModalFooter([
                    html.Div([
                        dbc.Button(id="learn-answer_a", color="secondary", n_clicks=0),
                        dbc.Button(id="learn-answer_b", color="secondary", n_clicks=0),
                        dbc.Button(id="learn-answer_c", color="secondary", n_clicks=0),
                    ], className="d-grid gap-2"),
                    html.Div([dbc.Button(children="Next question", id="next", color="primary", n_clicks=0)]),
                    html.Div(children='Score : 0/0', id='score'),
                    html.Div(children='Nb of errors : 0', id='errors')
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
     Output("errors", "children"),
     Output("session-history", "data")],
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
    global answer_save, answer_1, answer_2, answer_3, successes, nb_questions, questions_answered, correct_answer, nb_errors, current_session_id
    user = cache.get('identity')
    if current_session_id is None:
        current_session_id = cache.get(SESSION_CACHE_KEY)
    if button_id == 'play' and not is_open:
        successes = 0
        nb_questions = 0
        questions_answered = {} # question answered is here to prevent repeating to display the same question in a session
        nb_errors = 0
        current_session_id = _start_session(user)
        cache.set(SESSION_CACHE_KEY, current_session_id)
        
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
        _update_session_score(user, current_session_id, successes, nb_questions)

        table_data = _get_session_rows(user)
        return True, title, question, answer_1, answer_2, answer_3, answer_1_color, answer_2_color, answer_3_color, True, True, True, 0, 0, 0, f'Score : {successes} / {nb_questions}', f'Nb of errors : {nb_errors}', table_data
    else:
        table_data = _get_session_rows(user)
        return True, title, question, answer_1, answer_2, answer_3, "secondary", "secondary", "secondary", None, None, None, 0, 0, 0, f'Score : {successes} / {nb_questions}', f'Nb of errors : {nb_errors}', table_data # return invert of the current state of the modal, and the modal parameters




navbar_height = '50px'


def get_layout():
    identity = cache.get('identity')
    if identity is not None:
        session_rows = _get_session_rows(identity)
        return html.Div([
            popup_qa,
            html.Div(navbar, style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'zIndex': 1000}),
            html.Div([
                dbc.Button("I want to learn!", id='play', n_clicks=0, color="primary"),
                html.Div(
                    dash_table.DataTable(
                        id='session-history',
                        columns=[
                            {"name": "Session start", "id": "started_at"},
                            {"name": "Score", "id": "score"}
                        ],
                        data=session_rows,
                        style_table={'maxHeight': '300px', 'overflowY': 'auto', 'width': '100%'},
                        style_cell={
                            'textAlign': 'center',
                            'padding': '6px',
                            'backgroundColor': 'white',
                            'color': 'black'
                        },
                        style_header={
                            'backgroundColor': '#333333',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        page_action='none'
                    ),
                    style={'marginTop': '20px', 'width': '100%', 'maxWidth': '500px'}
                )

            ],
                style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'minHeight': '100vh',
                    'paddingTop': navbar_height,
                    'gap': '1rem'
                }),
        ])
    else:
        return dcc.Link('Go to login', href='/login')
        

layout = dcc.Link('Go to login', href='/login')

