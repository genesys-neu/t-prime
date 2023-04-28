import ast
import time
import os
import pandas as pd
#import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from queue import Queue
#from streamlit_autorefresh import st_autorefresh

PROTOCOLS = ['802_11ax', '802_11b', '802_11n', '802_11g']
COLORS = {'802_11ax': "#D81B60", '802_11b': "#FFC107", '802_11n': "#1E88E5", '802_11g': "#004D40"}
EMOJIS = {'802_11ax': '🟥', '802_11b': '🟦', '802_11n': '🟨', '802_11g': '🟩'}

def get_data():
    with open('output.txt', 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        last_line = last_line.replace('\n', '')
    return last_line

if "labels" not in st.session_state:
    st.session_state['labels'] = Queue(20) 
st.set_page_config(layout="wide", page_title="Real-Time ML Inference", page_icon=":wolf:",)
st.title('Transmitted protocol display')
st.write('In this dashboard, we will display the result of the real time classification from our ML module. This will be detecting \
          the protocol being transmitted among the following classes: 802_11ax, 802_11b, 802_11n, 802_11g.')
st.header('Protocols')

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

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader(f"{PROTOCOLS[0]}  {EMOJIS['802_11ax']}")
with c2:
    st.subheader(f"{PROTOCOLS[1]}  {EMOJIS['802_11b']}")
with c3:
    st.subheader(f"{PROTOCOLS[2]}  {EMOJIS['802_11n']}")
with c4:
    st.subheader(f"{PROTOCOLS[3]}  {EMOJIS['802_11g']}")

st.header('Real time prediction')

placeholder = st.empty()

while True:
    label = get_data()
    st.session_state.labels.put(label)
    time.sleep(0.5)
    with placeholder.container():
        st.markdown(f'<nobr class="extreme-font"> {EMOJIS[label]}  </nobr> <nobr class="big-font">{label}</nobr>', unsafe_allow_html=True)

        st.header('Prediction history')

