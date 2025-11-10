import ast
import time
import os
import json
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import argparse
import streamlit as st
#from paramiko import SSHClient
#from scp import SCPClient

##############################################
############### SSH CONNECTION ###############
##############################################
# Load connection credentials
# try:
#     with open('credentials.json') as f: 
#         creds = json.load(f)
# except FileNotFoundError:
#     print('File credentials.json not found! Please make sure to define it before calling this program.')
# Open ssh and scp connections
# ssh_ob = SSHClient()
# ssh_ob.load_system_host_keys()
# ssh_ob.connect(creds['host'], username=creds['username'], password=creds['password'])
# scp = SCPClient(ssh_ob.get_transport())
# Define the name of the file to extract and where to save
filename = 'output.txt' #creds['output_filename']
cwd = os.getcwd()

#####################################################
############### DISPLAY CONFIGURATION ###############
#####################################################
# Option to include background class
back_class = True #creds['background_class'] == "True" 
print("The model loaded has a background class:", back_class)
# Index Colors by label
PROTOCOLS_MAP = {'0':'802_11ax', '1':'802_11b', '2':'802_11n', '3':'802_11g', '4': 'Not known'}
PROTOCOLS = ['802_11ax', '802_11b', '802_11n', '802_11g', 'noise']
COLORS = ["#F20505", "#056CF2", "#FFCF00", "#0ABF04", "#000000"]
EMOJIS = ['AX', 'B', 'N', 'G', 'Not known'] #['🟥', '🟦', '🟨', '🟩'] 
sns.set(font_scale=2)

##############################################
############### DATA RETRIEVAL ###############
##############################################
def find_closest(lst, k):
    for i, num in enumerate(lst):
        if num >= k and i+1 != len(lst):
            
            if num > k:
                return i
            return i + 1 # We already had that sample
    # On last file read, there has not been any update')
    return None
        
def get_data(last_timestamp):
    #scp.get(os.path.join(creds['path_to_file'], filename), cwd)
    with open(filename, 'r') as f:
        ## FIND LAST LINE IMPLEMENTATION
        # try:  # catch OSError in case of a one line file 
        #     f.seek(-2, os.SEEK_END)
        #     while f.read(1) != b'\n':
        #         f.seek(-2, os.SEEK_CUR)
        # except OSError:
        #     f.seek(0)
        # last_line = f.readline().decode()
        # last_line = last_line.replace('\n', '')
        ## TIMESTAMPS APPROACH
        lbs, times = [], []
        for line in f.readlines():
            parts = line.split(' ')
            lbs.append(int(parts[0]))
            times.append(float(parts[1][:-2]))
        ix = find_closest(times, last_timestamp)
    if ix is None: # No update has been done

        return ix, last_timestamp
    return lbs[ix:], times[-1]

###########################################
############### WEBPAGE GUI ###############
###########################################
# Save predictions history
labels = []
last_timestamp = 0.0
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
if back_class: 
    c1, c2, c3, c4, c5 = st.columns(5)
else:
    c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader(f"{PROTOCOLS[0]}    -    {EMOJIS[0]}")
    st.image(f'{PROTOCOLS[0]}.png')
with c2:
    st.subheader(f"{PROTOCOLS[1]}    -    {EMOJIS[1]}")
    st.image(f'{PROTOCOLS[1]}.png')
with c3:
    st.subheader(f"{PROTOCOLS[2]}    -    {EMOJIS[2]}")
    st.image(f'{PROTOCOLS[2]}.png')
with c4:
    st.subheader(f"{PROTOCOLS[3]}    -    {EMOJIS[3]}")
    st.image(f'{PROTOCOLS[3]}.png')
if back_class:
    with c5:
        st.subheader(f"{EMOJIS[4]}")
        st.image(f'{PROTOCOLS[4]}.png')
# Real time updated dashboard
st.header('Real time prediction')
placeholder = st.empty()
while True:
    try:
        # Get new data
        plt.close()
        new_labels = None
        while new_labels is None:
            new_labels, last_timestamp = get_data(last_timestamp)
        # Display new data
        with placeholder.container():      
            labels.extend(new_labels)
            c1, c2, _ = st.columns(3)
            with c1:
                st.image(f'{PROTOCOLS[labels[-1]]}_big.png', width=450)
            with c2:
                st.markdown(f'<nobr class="extreme-font"> {EMOJIS[labels[-1]]}  </nobr>', unsafe_allow_html=True)
            #st.markdown(f'<nobr class="extreme-font"> {EMOJIS[label]}  </nobr> <nobr class="big-font">{PROTOCOLS[label]}</nobr>', unsafe_allow_html=True)
            st.header('Prediction history')
            #Display historic visualization of last predictions
            fig, ax = plt.subplots(figsize = (20, 2))
            if len(labels) > 300: 
                # Display only last 200
                ax = sns.heatmap([labels[-300:]], xticklabels=False, yticklabels=False, 
                                cmap=COLORS, cbar=False, vmin=0, vmax=4)
            else:
                ax = sns.heatmap([labels], xticklabels=False, yticklabels=False, 
                                    cmap=COLORS, cbar=False, vmin=0, vmax=4)
            ax.set_xlabel('Time')
            st.pyplot(fig)
        
    except KeyboardInterrupt:
        # When interrupting the infinite loop, close connections
        # scp.close()
        # ssh_ob.close()
        print('Connection closed, program terminated')
