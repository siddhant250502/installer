import streamlit as st
import pandas as pd
from sklearn.tree import export_graphviz
from streamlit_extras.stylable_container import stylable_container

def predictor_tree():
    if 'val' not in st.session_state:
        st.session_state['val'] = [st.session_state['filter_df'][i].min() for i in st.session_state['filter_df'].columns[1:]]
    # if st.session_state.reg:
    try:
        model = st.session_state['model']
        dot_data = export_graphviz(model.estimators_[0], out_file=None,
                            feature_names=st.session_state['filter_df'].columns[:-1],
                            class_names=[str(i) for i in st.session_state['filter_df'][st.session_state['filter_df'].columns[-1]].unique()],
                            filled=True, rounded=True,
                            special_characters=True)
        # try:
        col1, col2 = st.columns([1,7]) 
        with col1:
            number_input_value = []
            slider_val = []
            for num,column in enumerate(st.session_state.indep_vars):
                number_input_value.append(st.number_input(column, st.session_state['filter_df'][column].min(), st.session_state['filter_df'][column].max()))
                slider_val.append(st.slider(column, st.session_state['filter_df'][column].min(), st.session_state['filter_df'][column].max(), value=number_input_value[num], label_visibility='collapsed'))
            st.session_state['checkpoint'] = slider_val
            # if st.button('Save Checkpoint'):
            # st.session_state['checkpoint'] = slider_val
        with col2:
            dd_arr = []
            for i in range(len(dot_data)):
                if dot_data[i:i+10] == 'fillcolor=':
                    dd_arr.append(dot_data[i+11:i+18])
            for i in dd_arr:
                dot_data = dot_data.replace(i,'white')
            samples = pd.DataFrame(data=[slider_val], columns=st.session_state['filter_df'].columns[:-1])
            decision_paths = model.estimators_[0].decision_path(samples).toarray()[0]
            string1 = dot_data.splitlines()
            str1 = []
            str2 = []
            samples1 = pd.DataFrame(data=[st.session_state['checkpoint']], columns=st.session_state['filter_df'].columns[:-1])
            with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                ):
                st.subheader('Initial Point')
            st.dataframe(samples1, hide_index=True)
            with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                ):
                st.subheader(f"Tree for {st.session_state.dep_vars[0]}")
            decision_paths1 = model.estimators_[0].decision_path(samples1).toarray()[0]
            str3 = []
            str4 = []
            for n,dec in enumerate(decision_paths):
            # st.write(dot_data.splitlines()[20])
                if dec == 1 and n < 10:
                    for i in range(len(string1)):
                        if string1[i][:3] == f"{n} [":
                            str1.append(string1[i])
                            string1[i] = string1[i].replace('white', '#90EE90')
                            str2.append(string1[i])
                elif dec == 1 and n >= 10:
                    for i in range(len(string1)):
                        if string1[i][:4] == f"{n} [":
                            str1.append(string1[i])
                            string1[i] = string1[i].replace('white', '#90EE90')
                            str2.append(string1[i])
            for i in range(len(str1)):
                dot_data = dot_data.replace(str1[i],str2[i])

            for n,dec in enumerate(decision_paths1):
                # st.write(dot_data.splitlines()[20])
                if dec == 1 and n < 10:
                    for i in range(len(string1)):
                        if string1[i][:3] == f"{n} [":
                            str3.append(string1[i])
                            if 'white' in string1[i]:
                                string1[i] = string1[i].replace('white', '#FF7F7F')
                            if '#90EE90' in string1[i]:
                                string1[i] = string1[i].replace('#90EE90', '#FF7F7F')
                            str4.append(string1[i])
                elif dec == 1 and n >= 10:
                    for i in range(len(string1)):
                        if string1[i][:4] == f"{n} [":
                            str3.append(string1[i])
                            if 'white' in string1[i]:
                                string1[i] = string1[i].replace('white', '#FF7F7F')
                            if '#90EE90' in string1[i]:
                                string1[i] = string1[i].replace('#90EE90', '#FF7F7F')
                            str4.append(string1[i])
            for i in range(len(str3)):
                dot_data = dot_data.replace(str3[i],str4[i])
            st.graphviz_chart(dot_data)
            for i in range(len(decision_paths)):
                if decision_paths[len(decision_paths)-1-i]==1:
                    last_node = len(decision_paths)-1-i
                    break
            for i in range(len(string1)):
                    if last_node<10 and string1[i][:3] == f"{last_node} [":
                        if st.session_state['model_type'] == 'regression':
                            pred = string1[i].split('<br/>')[-1][:11]
                        else:
                            pred = string1[i].split('<br/>')[-1][:9]
                    elif last_node>=10 and string1[i][:4] == f"{last_node} [":
                        if st.session_state['model_type'] == 'regression':
                            pred = string1[i].split('<br/>')[-1][:11]
                        else:
                            pred = string1[i].split('<br/>')[-1][:9]
            with col2:
                with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                ):
                    st.subheader("Predictions")
                    st.info(f'Predicted {st.session_state.dep_vars[0]} {pred}')
    except:
        pass
