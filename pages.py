import streamlit as st
import pandas as pd
import os
import datetime
from data_info import data_analysis
from ml_model import ml_model_analysis
from decision_tree import decision_tree_viz
from predictor import predictor_tree
from streamlit_extras.stylable_container import stylable_container
from functions import nextpage, back

def page_1():
    st.title('List of Datasets')
    cwd = os.getcwd()
    files = os.listdir(cwd)
    documents = [f for f in files if os.path.isfile(os.path.join(cwd, f)) and f[-3:] == 'csv']
    data = pd.DataFrame({'Select': [False for i in range(len(documents))],
            'Index': [i+1 for i in range(len(documents))],
            'File Name': documents,
            'Timestamp': [datetime.datetime.fromtimestamp(os.path.getatime(os.path.join(cwd, f))) for f in documents],
            'File Size': [str((os.stat(f).st_size)//1024+1)+" KB" for f in documents]
    })

    res = st.data_editor(data,
                        column_config={"Select": st.column_config.CheckboxColumn(default=False), 
                                        "File Name": st.column_config.Column(width="large"), 
                                        "Timestamp": st.column_config.Column(width="large"),
                                        "File Size": st.column_config.Column(width="small")},
                        hide_index=True, 
                        use_container_width=True,
                        height=350,
                        )
    st.session_state['file_path'] = res.loc[res.Select.idxmax()]['File Name']
    with stylable_container(
        key='btn',
        css_styles=["""
            button {
                
                background-color: #30B9EF;
                color: #fff;
            }
            """,
            """
                button:hover {
                    background-color: #fff;
                    color: #30B9EF;
                }
            """,]
    ):
        upload_file = st.file_uploader("Upload your dataset", type=['csv'])
    if upload_file is not None:
        if upload_file.name[-3:] == 'csv':
            pd.read_csv(upload_file).to_csv(upload_file.name)
        else:
            pd.read_excel(upload_file).to_excel(upload_file.name)
        st.session_state['file_path'] = upload_file.name
    
    
    if len(res[res.Select == True])==1 or upload_file is not None:
        col7, col8, c9 = st.columns([1,0.055,0.07])
        with col8:
            st.button("Next", on_click=nextpage, disabled=(st.session_state.page > 1))
        with c9:
            if st.button("Delete"):
                if os.path.isfile(st.session_state['file_path']):
                    ind = res.index[res['File Name']==st.session_state['file_path']]
                    os.remove(st.session_state['file_path'])
                    st.rerun()
                else:
                    st.warning("File doesn't exist")
    else:
        st.write('Please select only 1 option / database')

def page_2():
    with stylable_container(
        key='btn',
        css_styles="""
            button {
                background: none!important;
                border: none;
                padding: 0!important;
                color: black;
                text-decoration: none;
                cursor: pointer;
                border: none !important;
                }
            button:hover {
                color: #30B9EF ;
            }
            button:focus {
                color: #30B9EF ;
                
            }
        """
    ):
        st.button('Back to Datasets', on_click=back)
    
    with st.container(border=True):
        t1, t2, t3, t4 = st.tabs(['Dataset Info','AI Model', 'Decision Tree Visualization', 'Predictor'])
        with t1:
            data_analysis()
        with t2:
            ml_model_analysis()
        with t3:
            decision_tree_viz()
        with t4:
            predictor_tree()
