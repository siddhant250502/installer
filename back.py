import streamlit as st
st.set_page_config(page_title="AI Model Analysis",layout="wide")

from pages import page_1, page_2
import toml

with open('styles/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {
    visibility: hidden;
    }
footer {
    visibility: hidden;
    }
</style>
"""
font = """
<style>
    body {
        font-family: 'Roboto' !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(font, unsafe_allow_html=True)

with open('.streamlit/config.toml', 'r') as f:
    config = toml.load(f)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = None

if 'btn_state' not in st.session_state:
    st.session_state['btn_state'] = False
    
if "page" not in st.session_state:
    st.session_state.page = 1
    
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = ' '

if 'reg' not in st.session_state:
    st.session_state['reg'] = False

if 'filter_df' not in st.session_state:
    st.session_state['filter_df'] = None
    
if 'data_header_df' not in st.session_state:
    st.session_state['data_header_df'] = None
    
if 'histogram_df' not in st.session_state:
    st.session_state['histogram_df'] = None
    
if 'corr_df' not in st.session_state:
    st.session_state['corr_df'] = None
    
if 'EXCLUDE_df' not in st.session_state:
    st.session_state['EXCLUDE_df'] = None
    
if 'col_name' not in st.session_state:
    st.session_state['col_name'] = set()
    
if "prev_tab" not in st.session_state:
    st.session_state['prev_tab'] = "Data Header"

if "model" not in st.session_state:
    st.session_state['model'] = 0
    
if "checkpoint" not in st.session_state:
    st.session_state['checkpoint'] = None
    
if "model_type" not in st.session_state:
    st.session_state['model_type'] = None

def main():
    if st.session_state['page'] == 1:
        page_1()
    elif st.session_state['page'] == 2:
        page_2()

if __name__ == '__main__':
    main()
