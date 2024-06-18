import streamlit as st
from sklearn.tree import export_graphviz

def decision_tree_viz():
    model = st.session_state['model']
    try:
        dot_data = export_graphviz(model.estimators_[0], out_file=None,
                        feature_names=st.session_state['filter_df'].columns[:-1],
                        class_names=[str(i) for i in st.session_state['filter_df'][st.session_state['filter_df'].columns[-1]].unique()],
                        filled=True, rounded=True,
                        special_characters=True)
        dd_arr = []
        for i in range(len(dot_data)):
            if dot_data[i:i+10] == 'fillcolor=':
                dd_arr.append(dot_data[i+11:i+18])
        for i in dd_arr:
            dot_data = dot_data.replace(i,'white')
        st.graphviz_chart(dot_data)
    except:
        pass