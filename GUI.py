import ast
import time
import os
import json
import pandas as pd
#import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from paramiko import SSHClient
from scp import SCPClient

#####################################################
############### DISPLAY CONFIGURATION ###############
#####################################################
# Index Colors by label
PROTOCOLS_MAP = {'0':'802_11ax', '1':'802_11b', '2':'802_11n', '3':'802_11g'}
PROTOCOLS = ['802_11ax', '802_11b', '802_11n', '802_11g']
COLORS = ["#D81B60", "#FFC107", "#1E88E5", "#004D40"]
EMOJIS = ['🟥', '🟦', '🟨', '🟩'] 

##############################################
############### SSH CONNECTION ###############
##############################################
# Load connection credentials
try:
    with open('credentials.json') as f: 
        creds = json.load(f)
except FileNotFoundError:
    print('File credentials.json not found! Please make sure to define it before calling this program.')
# Open ssh and scp connections
ssh_ob = SSHClient()
ssh_ob.load_system_host_keys()
ssh_ob.connect(creds['host'], username=creds['username'], password=creds['password'])
scp = SCPClient(ssh_ob.get_transport())
# Define the name of the file to extract and where to save
filename = creds['output_filename']
cwd = os.getcwd()

##############################################
############### DATA RETRIEVAL ###############
##############################################
def get_data():
    scp.get(os.path.join(creds['path_to_file'], filename), cwd)
    with open(filename, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        last_line = last_line.replace('\n', '')
    return last_line

###########################################
############### WEBPAGE GUI ###############
###########################################
# Save predictions history
labels = []
# Page config
st.set_page_config(layout="wide", page_title="Real-Time ML Inference", page_icon=":wolf:")
st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
<style>
.extreme-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)
# Main header and explanation
st.title('Transmitted protocol display')
st.write('In this dashboard, we will display the result of the real time classification from our ML module. This will be detecting \
          the protocol being transmitted among the following classes: 802_11ax, 802_11b, 802_11n, 802_11g.')
# Protocols legend
st.header('Protocols')
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader(f"{PROTOCOLS[0]}  {EMOJIS[0]}")
with c2:
    st.subheader(f"{PROTOCOLS[1]}  {EMOJIS[1]}")
with c3:
    st.subheader(f"{PROTOCOLS[2]}  {EMOJIS[2]}")
with c4:
    st.subheader(f"{PROTOCOLS[3]}  {EMOJIS[3]}")
# Real time updated dashboard
st.header('Real time prediction')
placeholder = st.empty()
while True:
    try:
        # Get new data
        label = int(get_data())
        labels.append(label)
        time.sleep(0.015)
        # Display new data
        with placeholder.container():
            st.markdown(f'<nobr class="extreme-font"> {EMOJIS[label]}  </nobr> <nobr class="big-font">{PROTOCOLS[label]}</nobr>', unsafe_allow_html=True)

            st.header('Prediction history')
            ## TODO: Display historic visualization of last predictions
    except KeyboardInterrupt:
        # When interrupting the infinite loop, close connections
        scp.close()
        ssh_ob.close()
        print('Connection closed, program terminated')
