import streamlit as st
import pandas as pd
from functions import regre
import numpy as np

def ml_model_analysis():
    if st.session_state['prev_tab']=='Data Header':
        if st.session_state['data_header_df'] is None:
            st.session_state['filter_df'] = pd.read_csv(st.session_state['file_path'])
        else:
            st.session_state['filter_df'] = st.session_state['data_header_df']
            
    elif st.session_state['prev_tab'] == 'Data Statistics':
        if st.session_state['histogram_df'] is None:
            st.session_state['filter_df'] = st.session_state['data_header_df']
        else:
            st.session_state['filter_df'] = st.session_state['histogram_df']
            
    elif st.session_state['prev_tab']=='Corelation Matrix':
        if st.session_state['corr_df'] is None:
            st.session_state['filter_df'] = st.session_state['histogram_df']
        else:
            st.session_state['filter_df'] = st.session_state['corr_df']
            
    elif st.session_state['prev_tab']=='Plot':
            st.session_state['filter_df'] = st.session_state['corr_df']
            
    st.title('AI Model Analysis')
    col4, col5, c6  = st.columns([1,1,0.2])
    col9, col10, col11, col12 = st.columns([0.4, 0.4, 0.3, 0.7])
    try:
        with col4.container(border=True):
            indep_vars = st.multiselect('Independent Variables', st.session_state['filter_df'].columns.values[:-1], placeholder = "Choose an option")
        with col5.container(border=True):
            dep_vars = st.multiselect('Dependent Variables', options=[x for x in st.session_state['filter_df'].columns.values[:-1] if x not in indep_vars], placeholder = "Choose an option", max_selections=1)
        unique_vals = len(np.unique(st.session_state['filter_df'][dep_vars]))
        if unique_vals<=10:  
            with col4.container(border=True):
                perf_reg = st.button('Classification')
        else:
            with col4.container(border=True):
                perf_reg = st.button('Regression')
        st.session_state.indep_vars, st.session_state.dep_vars = indep_vars, dep_vars
        if len(indep_vars)>=1 and len(dep_vars)>=1:
            st.session_state['filter_df']=st.session_state['filter_df'][st.session_state.indep_vars + st.session_state.dep_vars]
            # data = st.session_state['filter_df'][indep_vars+dep_vars]

        if perf_reg:
            regre(st.session_state['filter_df'], st.session_state.indep_vars, st.session_state.dep_vars)
            st.session_state.reg = True
        
    except AttributeError:
        pass
    except NameError:
        if len(indep_vars) == 0:
            st.warning('Choose Independent variable')
        if len(dep_vars) == 0:
            st.warning('Choose Dependent variable')