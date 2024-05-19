import streamlit as st
import requests
import toml
from io import BytesIO
import json


import streamlit as st
import toml
import requests
import matplotlib.pyplot as plt

# Config loading part : MAYBE USELESS FUNCTION
def load_config_streamlit(file_path="./.streamlit/config.toml"):
    try:
        config = toml.load(file_path)
        
        page_config = {
            "page_title": config["app"].get("title", None),
            "layout": config.get("layout", "wide"),
            "initial_sidebar_state": config.get("initial_sidebar_state", "auto")
        }

        # Appliquer la configuration de la page
        st.set_page_config(
            page_title=page_config["page_title"],
            layout=page_config.get("layout", "wide"),
            initial_sidebar_state=page_config.get("initial_sidebar_state", "auto"),
            page_icon=f":{config['app']['icon']}:"
        )

        return config, page_config

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier de configuration : {str(e)}")
        return None, None
    
def load_config_hyperparam(filepath="./parameters/frontend_parameters.json"):
    # Load the hyperparameters from the json file
    try:
        with open(filepath, "r", encoding="utf-8") as config_file:
            config_data = json.load(config_file)
            return config_data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier de configuration : {str(e)}")
        return None