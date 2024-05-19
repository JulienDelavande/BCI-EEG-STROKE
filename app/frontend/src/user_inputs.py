import streamlit as st
import requests
import toml
from io import BytesIO

import streamlit as st
import toml
import requests
import matplotlib.pyplot as plt

from datetime import timedelta, datetime, time
import numpy as np

# Update selection field in session state with deleted electrodes
def update_electrode_selection_input(removed_electrodes=None,validation_selection_button=False,reset_selection_button=False,frontend_parameters=None):
        
        if validation_selection_button and removed_electrodes is not None:
            st.session_state['selection'] = []
            st.session_state['selection'].append(frontend_parameters['parameters_tab']['selection_container']['selection_status'])
            for electrode in removed_electrodes:
                st.session_state['selection'].append({"field": "Sélection des données", "value": electrode})
            st.session_state['input_schema']['bad_channels'] = removed_electrodes
        if reset_selection_button:
            st.session_state['selection'] = []
            for key, value in frontend_parameters['initial_state']['selection'].items():
                st.session_state['selection'].append({"field": f"**{key}**", "value": f"{value}"})
            st.session_state['input_schema']['bad_channels'] = frontend_parameters['initial_state']['input_schema']['bad_channels']
        return None
    
def update_temporal_selection_input(start_time=None,end_time=None,validation_temporal_button=False,reset_temporal_button=False,frontend_parameters=None,no_crop=False):
        
        if validation_temporal_button and start_time is not None and end_time is not None:
            st.session_state['temporal'] = []
            st.session_state['temporal'].append(frontend_parameters['parameters_tab']['temporal_container']['temporal_selected_status'])
            st.session_state['temporal'].append({"field": "Sélection temporelle", "value": f"{start_time} et {end_time}"})
            # Convert time from datetime.time to float
            start_time_float = start_time.hour*3600.0 + start_time.minute*60.0 + start_time.second*1.0
            end_time_float = end_time.hour*3600.0 + end_time.minute*60.0 + end_time.second*1.0
            if no_crop :
                st.session_state['input_schema']['time_range']= [start_time_float, None].copy()
            else:
                st.session_state['input_schema']['time_range']= [start_time_float, end_time_float].copy()
        
        if reset_temporal_button:
            st.session_state['temporal'] = []
            for key, value in frontend_parameters['initial_state']['temporal'].items():
                st.session_state['temporal'].append({"field": f"**{key}**", "value": f"{value}"})
            st.session_state['input_schema']['time_range']=frontend_parameters['initial_state']['input_schema']['time_range']
        return None
    
def update_model_selection_input(model_selection=None,validation_model_button=False,reset_model_button=False,frontend_parameters=None):
    
    if validation_model_button and model_selection is not None:
        st.session_state['model'] = []
        st.session_state['model'].append(frontend_parameters['parameters_tab']['model_container']['model_selected_status'])
        st.session_state['model'].append({"field": "Modèle sélectionné", "value": model_selection})
        
        st.session_state['input_schema']['model'] = model_selection

    if reset_model_button:
        st.session_state['model'] = []
        for key, value in frontend_parameters['initial_state']['model'].items():
            st.session_state['model'].append({"field": f"**{key}**", "value": f"{value}"})
        st.session_state['input_schema']['model'] = frontend_parameters['initial_state']['input_schema']['model']

    return None
    
def slider_time_manager(backend_response_upload=None,frontend_parameters=None):
    if backend_response_upload is not None and frontend_parameters is not None:
        
        duration = 0
        if backend_response_upload['stroke_side'] == 'D':
            duration = backend_response_upload['signal_left_arm']['duration']  
        else:
            duration = backend_response_upload['signal_right_arm']['duration']
                
        if duration != 0:
            end_min, end_sec = divmod(np.floor(duration),60)
            start_time = time(hour=0,minute=0,second=0)
            end_time = time(hour=0,minute=int(end_min),second=int(end_sec))

            # Create the time range for slider options
            # Define the step size in seconds
            step = timedelta(seconds=frontend_parameters['parameters_tab']['temporal_container']['time_step'])

            # Create the time range for slider options
            time_range = []
            current_time = datetime(2021, 1, 1, start_time.hour, start_time.minute, start_time.second)
            while current_time <=datetime(2021,1,1,end_time.hour,end_time.minute,end_time.second) :
                time_range.append(time(current_time.hour, current_time.minute, current_time.second))
                current_time += step
    
            return (start_time, end_time, time_range)
    else:
        return None
   
def update_detection_input(backend_response=None,frontend_parameters=None):
    if backend_response is not None and frontend_parameters is not None:
        st.session_state['current_detection'] = frontend_parameters['detection_tab']['detection_success']
        st.session_state['last_detection'] = {}
        st.session_state['last_detection']['response'] = backend_response.copy()
        st.session_state['last_detection']['input_schema'] = st.session_state['input_schema'].copy()
    return None
