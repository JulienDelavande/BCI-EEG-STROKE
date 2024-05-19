import streamlit as st
import requests
import toml
from io import BytesIO

import streamlit as st
import toml
import requests
import matplotlib.pyplot as plt
import base64

from datetime import timedelta, datetime

def handle_uploaded_file(uploaded_file=None, input_schema_initialized=False):
    # Send the file to the backend : post a request
    if uploaded_file is not None:
        print(f'File uploaded: {uploaded_file.name}')
        response = requests.post("http://127.0.0.1:8000/upload/", files={"file": uploaded_file})
        # Parse the response to get the info we need for the container
        response = response.json()
        info_dict = {
            "Fichier sélectionné" : uploaded_file.name,
            "Patient" : response["patient_id"],
            "Numéro de session" : response["session_id"],
            "Côté parétique" : response["stroke_side"],
        }
        if not input_schema_initialized:
            st.session_state['input_schema']['brain_side'] = response["stroke_side"]
            if response["stroke_side"] == 'D':
                st.session_state['input_schema']['arm_side'] = 'G'
                st.session_state['input_schema']['time_range'] = [0.0, response['signal_left_arm']['duration']]
            else:
                st.session_state['input_schema']['arm_side'] = 'D'
                st.session_state['input_schema']['time_range'] = [0.0, response['signal_right_arm']['duration']]
            st.session_state['input_schema']['model'] = response["models_available"][0]
            st.session_state['input_schema']['data_location'] = response["data_location"] 
        
        return response, info_dict
    
    else : 
        return None
    
def handle_image(frontend_parameters=None):
    # If the stroke side is right
    if st.session_state['backend_response_upload']['stroke_side'] == 'D':
        electrodes_layout = st.session_state['backend_response_upload']['signal_left_arm']['electrodes']
        base64_image = st.session_state['backend_response_upload']['signal_left_arm'].get('electrodes_right_image')

        if base64_image is not None:
            # Affichez l'image
            image_selection = base64.b64decode(base64_image)
            return image_selection, electrodes_layout
    # If the stroke side is left
    else:
        electrodes_layout = st.session_state['backend_response_upload']['signal_right_arm']['electrodes']        
        base64_image = st.session_state['backend_response_upload']['signal_right_arm'].get('electrodes_left_image')
        if base64_image is not None:
            image_selection = base64.b64decode(base64_image)
            return image_selection, electrodes_layout
    return None, None

def handle_prediction(input_schema=None,frontend_parameters=None):
    # Send the file to the backend : post a request
    if input_schema is not None:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_schema)
        status_code = response.status_code
        response = response.json()
        if status_code == 200:
            return response
        else:
            if response is not None:
                st.error(frontend_parameters['detection_tab']['detection_bad_request'] + str(status_code))
            else:
                st.error(frontend_parameters['detection_tab']['detection_bad_request'] + "Error in the response")
    else:
        st.error(frontend_parameters['detection_tab']['detection_bad_input'] + "No input schema")
        return None
    
def handle_prediction_image(detection_response=None,frontend_parameters=None):
    if detection_response is not None:
        base64_image = detection_response.get('prediction_density_plot')

        if base64_image is not None:
            # Affichez l'image
            image_density = base64.b64decode(base64_image)
            return image_density
    return None
