import streamlit as st
import requests
import base64
import toml
import json
import streamlit as st
import toml
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import timedelta, datetime, time
from src.handle import handle_uploaded_file, handle_image, handle_prediction, handle_prediction_image
from src.session_state_management import init_session_state, update_session_state, reset_detection_state
from src.config_loader import load_config_hyperparam, load_config_streamlit
from src.user_inputs import update_electrode_selection_input, update_temporal_selection_input, slider_time_manager, update_model_selection_input, update_detection_input

# Config and frontend parameters loading
config, page_config = load_config_streamlit()
frontend_parameters = load_config_hyperparam()

# Session state initialization
if 'initialized' not in st.session_state:
    init_session_state(frontend_parameters=frontend_parameters, label='all')
    st.session_state['initialized'] = True

# -------------------------------------------------------------------
# Page design beginning
st.title(frontend_parameters['app_title'])

# Information sidebar
with st.sidebar:
    
    # Initialize upload container
    st.header(frontend_parameters['sidebar']['header_upload'])
    upload_container = st.container(border=True)
    
    # Update the upload container with the session state : placeholder if no files, file uploader if file not None
    with upload_container:
        uploaded_file = st.file_uploader(label=frontend_parameters['sidebar']['uploader_label'], type=["mat", "npy"])
        if uploaded_file is not None:
            if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != uploaded_file:
                if 'input_schema_initialized' not in st.session_state or st.session_state['input_schema_initialized'] == False:
                    st.session_state['input_schema_initialized'] = True
                    backend_response_upload,info_dict = handle_uploaded_file(uploaded_file=uploaded_file,input_schema_initialized=False)
                    update_session_state(backend_response=backend_response_upload,response_label='backend_response_upload',backend_response_dict=info_dict, state_label="info", reset=False)
                else:
                    backend_response_upload,info_dict = handle_uploaded_file(uploaded_file=uploaded_file,input_schema_initialized=True)
                    update_session_state(backend_response=backend_response_upload,response_label='backend_response_upload',backend_response_dict=info_dict, state_label="info", reset=False)
                # Update session state to avoid multiple request postings
                st.session_state['uploaded_file'] = uploaded_file        
        else:
            st.session_state['input_schema_initialized'] = False
            update_session_state(backend_response_dict=None,state_label="info", reset=True)
            st.session_state['uploaded_file'] = None
    
    # Initialize information container
    st.header(frontend_parameters['sidebar']['header_info'])
    info_container = st.container(border=True)
    # Update the information container with the session state : placeholder if no files, file info if file not None
    if st.session_state['info']:
        with info_container:
            for i in range(len(st.session_state['info'])):
                st.write(st.session_state['info'][i]["field"] + " : " + st.session_state['info'][i]["value"])    
    
    # Tools for developers and researchers
    st.header(frontend_parameters['sidebar']['header_tools'])
    
    # Toggle for help expanders in the main page
    help_toggle = st.toggle(frontend_parameters['sidebar']['help_toggle_label'],key="help_toggle",value=True)
    
    # Backend health check
    test_button = st.button(frontend_parameters['button_label']['health_label'],use_container_width=True,key="test_button_")
    
# -------------------------------------------------------------------
# Main page code 

# Initialize the tabs
tab1, tab2, tab3, tab4 = st.tabs(frontend_parameters['tabs']['labels'])

# -------------------------------------------------------------------
# Parameters tab
with tab1:
    
    # -------------------------------------------------------------------
    # Section for parameters linked to the experiment 
    st.header(frontend_parameters["parameters_tab"]["expe_parameters_header"])
    
    # Initialize the containers for experimental parameters
    selection_container = st.container(border=True)
    temporal_container = st.container(border=True)
    
    # Electrodes selection container
    with selection_container:
        st.header(frontend_parameters['parameters_tab']['selection_container']['header'])
        select_col1, select_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
        
        # Util column
        with select_col1:
            # Help expander
            if help_toggle:
                with st.expander(frontend_parameters['help_expander']['help_label']):
                        for help_message in frontend_parameters['parameters_tab']['selection_container']['help_content']:
                            st.write(help_message)
            
            # Status display 
            select_status_col = st.container(border=True)
            
            # Buttons display
            #st.subheader(frontend_parameters['parameters_tab']['buttons_subheader'])
            reset_selection_button = st.button(frontend_parameters['button_label']['reset_label'],use_container_width=True,key="reset_selection")
            validation_selection_button = st.button(frontend_parameters['button_label']['validate_label'],use_container_width= True,key="validation_selection")
    
        # Initialize display column
        with select_col2:
                display_selection_col = st.container(border=True)
                with display_selection_col:
                    st.subheader(frontend_parameters['parameters_tab']['display_col']['subheader'])
           
        # If file is uploaded, display the electrodes signal
        if uploaded_file is not None:
            with select_col2:
                with display_selection_col:
                    image_selection, electrodes_layout = handle_image(frontend_parameters=frontend_parameters)
                    if image_selection is not None:
                        st.image(image_selection)
                    else:
                        st.error(frontend_parameters['parameters_tab']['selection_container']['no_image_error'])
            
            # User widget for electrodes selection
            selection_label = frontend_parameters['parameters_tab']['selection_container']['selection_label']
            if electrodes_layout is not None:
                removed_electrodes = st.multiselect(selection_label,electrodes_layout)
                # Update the session state with the user input
                update_electrode_selection_input(removed_electrodes=removed_electrodes,validation_selection_button=validation_selection_button,reset_selection_button=reset_selection_button,frontend_parameters=frontend_parameters)
            else :
                st.error(frontend_parameters['parameters_tab']['selection_container']['no_electrode_error'])
        
        else :
            with display_selection_col:
                st.warning(frontend_parameters['parameters_tab']['selection_container']['no_file_warning'])

    # Beginning and end of the experiment selection container
    with temporal_container:
        st.header(frontend_parameters['parameters_tab']['temporal_container']['header'])
        temporal_col1, temporal_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
        
        # Status column
        with temporal_col1:
            # Help expander
            if help_toggle:
                with st.expander(frontend_parameters['help_expander']['help_label']):
                    for help_message in frontend_parameters['parameters_tab']['temporal_container']['help_content']:
                        st.write(help_message)
            
            temporal_status_col = st.container(border=True)
        
            #st.subheader(frontend_parameters['parameters_tab']['buttons_subheader'])
            reset_temporal_button = st.button(frontend_parameters['button_label']['reset_label'],use_container_width=True,key="reset_temporal")
            validation_temporal_button = st.button(frontend_parameters['button_label']['validate_label'],use_container_width= True,key="validation_temporal")
                  
        # Initialize display column for temporal selection container
        with temporal_col2:
            display_temporal_col = st.container(border=True)
            with display_temporal_col:
                st.subheader(frontend_parameters['parameters_tab']['display_col']['subheader'])
                
        # Previous condition to be matched for this input
        if uploaded_file is not None:
            if image_selection is not None:
                with display_temporal_col:
                    st.image(image_selection)
            else:
                with display_temporal_col:
                    st.error(frontend_parameters['parameters_tab']['temporal_container']['no_image_error'])
            # Slider label from json parameter
            slider_label = frontend_parameters['parameters_tab']['temporal_container']['temporal_label']
            
            # Define the range of the slider in seconds
            time_slider_results = slider_time_manager(backend_response_upload=st.session_state['backend_response_upload'],frontend_parameters=frontend_parameters) 
            
            if time_slider_results is not None:
                start_time, end_time, time_range = time_slider_results
                # Slider widget
                start_time_exp, end_time_exp = st.select_slider(
                    slider_label,
                    options=time_range,
                    value=(start_time,end_time),
                )
                
                # Update the session state with the user input
                if end_time_exp == end_time:
                    update_temporal_selection_input(start_time=start_time_exp,end_time=end_time_exp,validation_temporal_button=validation_temporal_button,reset_temporal_button=reset_temporal_button,frontend_parameters=frontend_parameters,no_crop=True)
                else :
                    update_temporal_selection_input(start_time=start_time_exp,end_time=end_time_exp,validation_temporal_button=validation_temporal_button,reset_temporal_button=reset_temporal_button,frontend_parameters=frontend_parameters,no_crop=False)

            else:  
            # Two errors to be handled here : no signal and no file uploaded
                st.error(frontend_parameters['parameters_tab']['temporal_container']['no_signal_error'])
        else :
            with display_temporal_col:
                st.warning(frontend_parameters['parameters_tab']['temporal_container']['no_file_warning'])
    
    # -------------------------------------------------------------------
    # Section for parameters of the pipeline
    st.header(frontend_parameters["parameters_tab"]["pipeline_parameters_header"])
    
    # Initialize the containers for pipeline parameters
    model_container = st.container(border=True)
    
    with model_container:
        st.header(frontend_parameters['parameters_tab']['model_container']['header'])
        model_col1, model_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
        
        # Status and buttons column
        with model_col1:
            # Help expander
            if help_toggle:
                with st.expander(frontend_parameters['help_expander']['help_label']):
                    for help_message in frontend_parameters['parameters_tab']['model_container']['help_content']:
                        st.write(help_message)
            model_status_col = st.container(border=True)
                 
            reset_model_button = st.button(frontend_parameters['button_label']['reset_label'],use_container_width=True,key="reset_model")
            validation_model_button = st.button(frontend_parameters['button_label']['validate_label'],use_container_width= True,key="validation_model")
                
        # Display column
        with model_col2:
            model_display_col = st.container(border=True)
            with model_display_col:
                st.subheader(frontend_parameters['parameters_tab']['model_container']['display_subheader'])
                
        # Condition to be matched 
        if uploaded_file is not None:
            model_details = st.session_state['backend_response_upload']['models_info']
            with model_display_col:
                for key, value in model_details.items():
                    st.markdown(f"{key} : {value['composition']}")
            model_selection_label = frontend_parameters['parameters_tab']['model_container']['model_label']
            model_options = st.session_state['backend_response_upload'].get('models_available')
            if model_selection_label is not None:
            # User input widget
                model_selection=st.selectbox(label=model_selection_label,
                                        options=model_options,
                                        index=None,key="model_selectbox", placeholder=model_selection_label)
                update_model_selection_input(model_selection=model_selection,validation_model_button=validation_model_button,reset_model_button=reset_model_button,frontend_parameters=frontend_parameters)
            else :
                st.error(frontend_parameters['parameters_tab']['model_container']['no_model_error'])
        else :
            with model_display_col:
                st.warning(frontend_parameters['parameters_tab']['model_container']['no_file_warning'])
# -------------------------------------------------------------------
# Movement detection tab
with tab2 :

    detection_container = st.container(border=True)
    with detection_container:
        st.header(frontend_parameters['detection_tab']['data_detection_container']['header'])
        detection_col1, detection_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
        
        # Status and buttons column
        with detection_col1:
            # Help expander
            if help_toggle:
                with st.expander(frontend_parameters['help_expander']['help_label']):
                    for help_message in frontend_parameters['detection_tab']['data_detection_container']['help_content']:
                        st.write(help_message)
            
            detection_status_col = st.container(border=True)
                
        # Display column
        with detection_col2:
            with st.container(border=True):
                st.subheader(frontend_parameters['detection_tab']['data_detection_container']['display_subheader'])      
                for key, value in frontend_parameters['detection_tab']['data_detection_container']['display_content'].items():
                    if key == 'time_range':
                        start_disp, end_disp = st.session_state['input_schema']['time_range']
                        if end_disp == None:
                            st.write(frontend_parameters['detection_tab']['no_crop_status'])
                        else:
                            start_min, start_sec = divmod(start_disp,60)
                            end_min, end_sec = divmod(end_disp,60)
                            st.write(f"{value} {time(hour=0,minute=int(start_min),second=int(start_sec))} et {time(hour=0,minute=int(end_min),second=int(end_sec))}")
                    else:
                        st.write(f"{value} {st.session_state['input_schema'][key]}")
        proc_label = frontend_parameters['detection_tab']['data_detection_container']['detection_button_label']
        
        # Condition to be matched 
        if uploaded_file is not None:
            # User input widget
            detection_button=st.button(proc_label,use_container_width=True,key="detection_button")
            if detection_button:
                detection_response = handle_prediction(input_schema=st.session_state['input_schema'],frontend_parameters=frontend_parameters)
                if detection_response is not None:
                        update_detection_input(backend_response=detection_response,frontend_parameters=frontend_parameters)
        else :
            reset_detection_state()
            st.warning(frontend_parameters['detection_tab']['data_detection_container']['no_file_warning'])

# -------------------------------------------------------------------
# Results tab
with tab3 :
    
    intention_container = st.container(border=True)
    
    with intention_container:
        st.header(frontend_parameters['results_tab']['intention_container']['header'])
        
        intention_col1, intention_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
        with intention_col1:
            # Help expander
            if help_toggle:
                with st.expander(frontend_parameters['help_expander']['help_label']):
                    for help_message in frontend_parameters['results_tab']['intention_container']['help_content']:
                        st.write(help_message)
            intention_info_col = st.container(border=True)
            
        with intention_col2:
            with st.container(border=True):
                st.subheader(frontend_parameters['parameters_tab']['display_col']['subheader'])
                if 'last_detection' in st.session_state and 'response' in st.session_state['last_detection']:
                    density_plot_image = handle_prediction_image(st.session_state['last_detection']['response'])
                    st.image(density_plot_image)
        if uploaded_file is None:
            st.warning(frontend_parameters['results_tab']['intention_container']['no_file_warning'])
            
# -------------------------------------------------------------------
# User guide tab
with tab4:
    welcome_container = st.container(border=True)
    with welcome_container:
        st.header(frontend_parameters['user_guide_tab']['welcome_container']['header'])
        
        intro_container = st.container(border=True)
        with intro_container :
            
            for intro_message in frontend_parameters['user_guide_tab']['welcome_container']['intro']:
                st.write(intro_message)
                
        toc_container = st.container(border=True)
        with toc_container:
            for toc_message in frontend_parameters['user_guide_tab']['welcome_container']['table_of_content']:
                st.markdown(toc_message)

        st.markdown(frontend_parameters['user_guide_tab']['welcome_container']['funny_message'])
    sidebar_container = st.container(border=True)
    with sidebar_container:
        st.header(frontend_parameters['user_guide_tab']['sidebar_container']['header'])
        for sidebar_message in frontend_parameters['user_guide_tab']['sidebar_container']['content']:
            st.markdown(sidebar_message)
        st.image(frontend_parameters['user_guide_tab']['sidebar_container']['image'])
        uploader_help_container = st.container(border=True)
        with uploader_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['sidebar_container']['uploader_help']['header'])
            uploader_help_col1, uploader_help_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
            with uploader_help_col1:
                st.image(frontend_parameters['user_guide_tab']['sidebar_container']['uploader_help']['image'])
            with uploader_help_col2:
                for uploader_message in frontend_parameters['user_guide_tab']['sidebar_container']['uploader_help']['content']:
                    st.markdown(uploader_message)
        information_help_container = st.container(border=True)
        with information_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['sidebar_container']['information_help']['header'])
            info_help_col1, info_help_col2 = st.columns(frontend_parameters['parameters_tab']['size_column'])
            with info_help_col1:
                st.image(frontend_parameters['user_guide_tab']['sidebar_container']['information_help']['image'])
            with info_help_col2:
                for information_message in frontend_parameters['user_guide_tab']['sidebar_container']['information_help']['content']:
                    st.markdown(information_message)
        tools_help_container = st.container(border=True)
        with tools_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['sidebar_container']['tools_help']['header'])
            for tools_message in frontend_parameters['user_guide_tab']['sidebar_container']['tools_help']['content']:
                st.markdown(tools_message)
    
    parameters_help_container = st.container(border=True)

    with parameters_help_container:
        st.header(frontend_parameters['user_guide_tab']['parameters_container']['header'])
        st.image(frontend_parameters['user_guide_tab']['parameters_container']['image'])
        for parameters_message in frontend_parameters['user_guide_tab']['parameters_container']['content']:
            st.markdown(parameters_message)
        st.subheader(frontend_parameters['user_guide_tab']['parameters_container']['subheader_expe'])
        electrodes_help_container = st.container(border=True)
        with electrodes_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['parameters_container']['electrodes']['header'])
            st.image(frontend_parameters['user_guide_tab']['parameters_container']['electrodes']['image'])
            for electrodes_message in frontend_parameters['user_guide_tab']['parameters_container']['electrodes']['content']:
                st.markdown(electrodes_message)
        temporal_help_container = st.container(border=True)
        with temporal_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['parameters_container']['temporal']['header'])
            st.image(frontend_parameters['user_guide_tab']['parameters_container']['temporal']['image'])
            for temporal_message in frontend_parameters['user_guide_tab']['parameters_container']['temporal']['content']:
                st.markdown(temporal_message)
        st.subheader(frontend_parameters['user_guide_tab']['parameters_container']['subheader_pipeline'])
        model_help_container = st.container(border=True)
        with model_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['parameters_container']['model']['header'])
            st.image(frontend_parameters['user_guide_tab']['parameters_container']['model']['image'])
            for model_message in frontend_parameters['user_guide_tab']['parameters_container']['model']['content']:
                st.markdown(model_message)
    
    detection_container = st.container(border=True)
    with detection_container:
        st.header(frontend_parameters['user_guide_tab']['detection_container']['header'])
        st.image(frontend_parameters['user_guide_tab']['detection_container']['image'])
        for detection_message in frontend_parameters['user_guide_tab']['detection_container']['content']:
            st.markdown(detection_message)
        detection_help_container = st.container(border=True)
        with detection_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['detection_container']['detection']['header'])
            st.image(frontend_parameters['user_guide_tab']['detection_container']['detection']['image1'])
            st.image(frontend_parameters['user_guide_tab']['detection_container']['detection']['image2'])
            for detection_help_message in frontend_parameters['user_guide_tab']['detection_container']['detection']['content']:
                st.markdown(detection_help_message)
    
    results_container = st.container(border=True)
    with results_container:
        st.header(frontend_parameters['user_guide_tab']['results_container']['header'])
        st.image(frontend_parameters['user_guide_tab']['results_container']['image'])
        for results_message in frontend_parameters['user_guide_tab']['results_container']['content']:
            st.markdown(results_message)
        results_help_container = st.container(border=True)
        with results_help_container:
            st.subheader(frontend_parameters['user_guide_tab']['results_container']['results']['header'])
            st.image(frontend_parameters['user_guide_tab']['results_container']['results']['image'])
            for results_help_message in frontend_parameters['user_guide_tab']['results_container']['results']['content']:
                st.markdown(results_help_message)
            
# -------------------------------------------------------------------
# Backend communication button
if test_button:
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        response = response.text
        with st.sidebar:
            st.success("L'application est prête à fonctionner! 🚀")
    except requests.exceptions.RequestException as e:
        with st.sidebar:
            st.error(f"⛔ Erreur lors de la requête avec le backend: {str(e)}")
            
# -------------------------------------------------------------------
# Session state display
with select_status_col:
    st.subheader(frontend_parameters['parameters_tab']['status_subheader'])
    if 'selection' in st.session_state and st.session_state['selection']:
        for i in range(len(st.session_state['selection'])):
            st.write(st.session_state['selection'][i]["value"])

with temporal_status_col:
    st.subheader(frontend_parameters['parameters_tab']['status_subheader'])
    if st.session_state['temporal']:
        for i in range(len(st.session_state['temporal'])):
            st.write(st.session_state['temporal'][i]["value"])

with model_status_col:
    st.subheader(frontend_parameters['parameters_tab']['status_subheader'])
    if st.session_state['model']:
        for i in range(len(st.session_state['model'])):
            st.write(st.session_state['model'][i]["value"])
             
with detection_status_col:
    st.subheader(frontend_parameters['detection_tab']['status_subheader'])
    if st.session_state['current_detection']:
        st.write(st.session_state['current_detection'])
        if 'last_detection' in st.session_state and 'response' in st.session_state['last_detection'] and 'input_schema' in st.session_state['last_detection']:
            for key, value in frontend_parameters['detection_tab']['data_detection_container']['display_content'].items():
                if key == 'time_range':
                    start_disp, end_disp = st.session_state['last_detection']['input_schema']['time_range']
                    if end_disp == None:
                        st.write(frontend_parameters['detection_tab']['no_crop_status'])
                    else:
                        start_min, start_sec = divmod(start_disp,60)
                        end_min, end_sec = divmod(end_disp,60)
                        st.write(f"{value} {time(hour=0,minute=int(start_min),second=int(start_sec))} et {time(hour=0,minute=int(end_min),second=int(end_sec))}")
                else:
                    st.write(f"{value} {st.session_state['last_detection']['input_schema'][key]}")

with intention_info_col:
    st.subheader(frontend_parameters['results_tab']['infos_subheader'])   
    if 'last_detection' in st.session_state and 'response' in st.session_state['last_detection'] and 'input_schema' in st.session_state['last_detection']:
        for key, value in frontend_parameters['results_tab']['intention_container']['display_content'].items():
            if key == "mvt_or_intention": 
                if "INT_MVT" in st.session_state['last_detection']['response']['processing_info']['model']:
                    st.write(f"{value} l'intention de mouvement.")
                else : 
                    st.write(f"{value} le mouvement.")
            else:
                if key == 'threshold':
                    st.write(f"{value} 0.5")
