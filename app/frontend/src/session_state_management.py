# -*- coding: utf-8 -*-

import streamlit as st
import requests
import toml
from io import BytesIO

import streamlit as st
import toml
import requests
import matplotlib.pyplot as plt


# Session state initializer 
# used first to initialize the session state, then to reinitalize it
def init_session_state(frontend_parameters=None,label='all'):
    if frontend_parameters is not None:
        # Dataset information container initial state
        if label=='all' or label=='info' :
            st.session_state['info'] = []
            for key, value in frontend_parameters['initial_state']['info'].items():
                st.session_state['info'].append({"field": f"**{key}**", "value": f"{value}"})
            
        # Electrode selection initial state
        if label=='all' or label=='selection' :
            st.session_state['selection']=[]
            for key, value in frontend_parameters['initial_state']['selection'].items():
                st.session_state['selection'].append({"field": f"**{key}**", "value": f"{value}"})

        # Temporal selection initial state
        if label=='all' or label=='temporal' :
            st.session_state['temporal']=[]
            for key, value in frontend_parameters['initial_state']['temporal'].items():
                st.session_state['temporal'].append({"field": f"**{key}**", "value": f"{value}"})

        # Model selection initial state
        if label=='all' or label=='model' :
            st.session_state['model']=[]
            for key, value in frontend_parameters['initial_state']['model'].items():
                st.session_state['model'].append({"field": f"**{key}**", "value": f"{value}"})
        # Input schema initial state
        if label=='all' or label=='input_schema' :
            st.session_state['input_schema'] = frontend_parameters['initial_state']['input_schema']
        # Detection initial state
        if label=='all' or label=='current_detection' :
            st.session_state['current_detection'] = frontend_parameters['initial_state']['current_detection']
        # Last detection initial state
        if label=='all' or label=='last_detection' :
            st.session_state['last_detection'] = frontend_parameters['initial_state']['last_detection']
    return st.session_state

# Session state updater
def update_session_state(backend_response=None,response_label='',backend_response_dict=None,state_label='',reset=True,frontend_parameters=None):
    # If the backend response is not None, update the session state with the backend response
    if backend_response_dict is not None and not reset:
        st.session_state[state_label] = []
        for key, value in backend_response_dict.items():
            st.session_state[state_label].append({"field": f"**{key}**", "value": f"{value}"})
    else:
        # If the backend response is None, reset the session state to its initial state
        if reset:
            init_session_state(frontend_parameters=frontend_parameters,label=state_label)
    if backend_response is not None:
        st.session_state[response_label] = backend_response
    return st.session_state

def reset_detection_state(frontend_parameters=None):
    init_session_state(frontend_parameters=frontend_parameters,label='current_detection')
    return st.session_state
