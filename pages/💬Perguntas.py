import streamlit as st

from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import sys
import time
from jobs_details import jobs_details as data
from dotenv import load_dotenv
import ast

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

st.title('Perguntas 💬')
st.write("histórico do chatbot.")

with open('mensagens.txt', 'r') as file:
    content = file.read()

st.write(content)