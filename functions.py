import streamlit as st
import pandas as pd
import statistics
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import time
import numpy as np
import math

def nextpage(): st.session_state.page += 1
def back(): 
    st.session_state.page -= 1 
    st.session_state.button_clicked = None
    st.session_state['col_name'] = set()
    st.session_state['filter_df'] = st.session_state['data_header_df'] = st.session_state['histogram_df'] = st.session_state['corr_df'] = st.session_state['EXCLUDE_df'] = st.session_state['checkpoint'] = st.session_state['model_type'] = None
    st.session_state['prev_tab'] = "Data Header"
placeholder = st.empty()

def r_square(df):
    dict_corr = {'Column 1': [], 'Column 2': [], 'R-Squared': []}#
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            lr = LinearRegression()
            lr.fit([[i] for i in df[df.columns[i]]], [[i] for i in df[df.columns[j]]])
            dict_corr['Column 1'].append(df.columns[i])
            dict_corr['Column 2'].append(df.columns[j])
            dict_corr['R-Squared'].append(lr.score([[i] for i in df[df.columns[i]]], [[i] for i in df[df.columns[j]]]))
    return(pd.DataFrame(dict_corr).sort_values(by = 'R-Squared', ascending=False))
    

def data_cleaning(data1):
    # Format columns
    for i in data1.columns:
        if type(data1[i]) == str:
            data1[i] = data1[i].str.replace(",","")
    # Removing columns with string data
    dt = []
    for num,i in enumerate(data1.dtypes):
        if i != "int64" and i != "float64":
            dt.append(num)
    data1.drop(data1.columns[dt],axis=1,inplace=True)
    
    # Dealing with Null values
    data1.fillna(0, inplace=True)
    # data1 = mice(data1)
    # dropping colummns with one uniques value
    for i in data1.columns:
        if len(data1[f'{i}'].unique()) <= 1:
            data1.drop(i, inplace=True, axis=1) 
        elif len(data1[f'{i}'].unique())==2  and True in np.isnan(data1[f'{i}'].unique()):
            data1.drop(i, inplace=True, axis=1) 
    return data1

def mice(df):
    # imputer = IterativeImputer(max_iter=10, random_state=21)
    # imputed_values = imputer.fit_transform(df)
    # df = pd.DataFrame(imputed_values, columns=df.columns)
    return df
    
def interactive_plot(df, x_axis, y_axis, ols):
    new_data = {True:'INCLUDE', False:'EXCLUDE'}
    df = df.replace({'EXCLUDE/INCLUDE':new_data})
    try:
        if x_axis and y_axis:
            if ols:
                fig = px.scatter(df[df['EXCLUDE/INCLUDE']=='INCLUDE'], x=x_axis, y=y_axis, color='EXCLUDE/INCLUDE', color_discrete_map={'INCLUDE':'#30B9EF'}, symbol='EXCLUDE/INCLUDE', symbol_map={'INCLUDE':'circle-open'}, trendline='ols', trendline_color_override='#000000', height=450, custom_data=[x_axis, y_axis]) 
                fig.update_traces(hovertemplate=None)
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color='EXCLUDE/INCLUDE', color_discrete_map={'EXCLUDE':'#D22B2B', 'INCLUDE':'#30B9EF'}, symbol='EXCLUDE/INCLUDE', symbol_map={'EXCLUDE':'x', 'INCLUDE':'circle-open'}, height=450, custom_data=[df[x_axis], df[y_axis]])    
        fig.update_layout(title_text='Column Vs Column Scatter Plot', legend_title_text='Data Points')
        return fig
    except Exception as e:
        st.error(e)
        
def std(df):
    x = 0
    mean = df.mean()
    for i in df:
        x+=(mean-i)**2
    stddev = math.sqrt(x/(len(df)-1))
    return stddev

def difference(x1):    
    return abs(x1.iloc[1]-x1.iloc[0]) 

def est_std(df, opt):
    d2 = {1:1.128, 2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.970, 10:3.078}
    df.columns = [i.upper() for i in df.columns]
    sg = {}
    for i in df.columns:
        if "(" in i and ")" in i:
            i=i[:i.index("(")]
        x=sg.get(i,0)
        sg[i]=x+1
    for i in sg.keys():
        if sg[i]>1:
            df[f'{i}_Avg'] = 0
            for j in df.columns[:-1]:
                if i in j:
                    df[f'{i}_Avg'] += df[j]
            df[f'{i}_Avg']=df[f'{i}_Avg']/sg[i]
    ma = df[opt.upper()].rolling(window=2).apply(difference)
    if "(" in opt and ")" in opt or'_avg' in opt:
        opt=opt[:-4]
    return (ma.mean()/d2[sg[opt.upper()]])

def plot_columns(df, opt, sigma, est_sigma, curve):
    new_data = {True:'INCLUDE', False:'EXCLUDE'}
    df = df.replace({'EXCLUDE/INCLUDE':new_data})
    mean = statistics.mean(df[opt])
    if sigma:
        stddev = std(df[opt])
    if est_sigma:
        stddev = est_std(df, opt)
        # cor = []
        # for i in df[opt]:
        #     cor.append((i-mean)**2)
        # stddev = math.sqrt(sum(cor)/len(df[opt])-1)
    fig = px.histogram(df, x=opt, color_discrete_map={'EXCLUDE':'#D22B2B', 'INCLUDE':'#30B9EF'}, color='EXCLUDE/INCLUDE')
    fig.update_layout(legend_title_text='Data Points')
    fig.update_traces(xbins_size= (df[opt].max()-df[opt].min())/math.sqrt(len(df[opt]))) 
    fig0 = px.histogram(df, x=opt, color_discrete_sequence=['#30B9EF'], histnorm = 'probability density')
    f = fig0.full_figure_for_development(warn=False)
    xbins = f.data[0].xbins
    plotbins = list(np.arange(start=xbins['start'], stop=xbins['end']+xbins['size'], step=xbins['size']))
    counts, bins = np.histogram(list(f.data[0].x), bins=plotbins)
    # st.write(bins)
    bins = [i+(df[opt].max()-df[opt].min())/math.sqrt(len(df[opt]))/2 for i in bins]
    if curve:
        fig1 = px.line(x=bins[:-1], y=counts)
        fig1.update_traces(line_width=0.9, patch = {"line_shape":"spline"})
        fig.add_trace(fig1.data[0])
    if sigma or est_sigma:
        one_sigma_plus = mean+(1*stddev)
        one_sigma_minus = mean-(1*stddev)
        two_sigma_plus = mean+(2*stddev)
        two_sigma_minus = mean-(2*stddev)
        three_sigma_plus = mean+(3*stddev)
        three_sigma_minus = mean-(3*stddev)
        fig.add_vline(mean, line_dash="dash",annotation_text="Mean", annotation_position="left")
        fig.add_vrect(x0=one_sigma_minus, x1=one_sigma_plus, annotation_text='1-sigma', fillcolor="#00FF00", opacity=0.2, line_width=0.5)
        # fig.add_vrect(x0=one_sigma_plus, x1=mean, annotation_text='one-sigma', fillcolor="green", opacity=0.25)
        fig.add_vrect(x0=two_sigma_minus, x1=one_sigma_minus, fillcolor="#FFFF00", opacity=0.2, line_width=0.5, annotation_text='2-sigma')
        fig.add_vrect(x0=two_sigma_plus, x1=one_sigma_plus,  fillcolor="#FFFF00", opacity=0.2, line_width=0.5)
        fig.add_vrect(x0=three_sigma_minus, x1=two_sigma_minus, fillcolor="purple", opacity=0.2, line_width=0.5)
        fig.add_vrect(x0=three_sigma_plus, x1=two_sigma_plus, annotation_text='3-sigma', fillcolor="#A020F0", opacity=0.2, line_width=0.5)
    if sigma and est_sigma:
        st.error('Please Select one')
        return px.bar(None)
    else:
        fig.update_layout(bargap=0.1)
        return fig
    
    
def correlation(df):
    res_df = pd.DataFrame(index=df.columns, columns=df.columns)
    for i in res_df.columns:
        for j in res_df.index:
            lr = LinearRegression()
            lr.fit([[i] for i in df[i]], [[i] for i in df[j]])
            res_df[i][j] = lr.score([[i] for i in df[i]], [[i] for i in df[j]])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=res_df, x=df.columns, y=df.columns, colorscale='RdBu_r',  hovertemplate='X-Value: %{x}<br>Y-Value: %{y}<br>R2: %{z}<extra></extra>'))
    fig.update_layout(modebar_remove=['zoom', 'pan'])
    return fig

def regre(df, indep_vars, dep_vars):
    start = time.time()
    df = data_cleaning(df)
    X = df[indep_vars]
    y = df[dep_vars[0]]
    # oversample = SMOTE()
    # undersample = 

    if len(y.unique()) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        rfr = RandomForestRegressor(max_depth = 5, max_leaf_nodes=12)
        st.session_state['model_type'] = 'regression'
        try:
            rfr.fit(X_train, y_train)
        except ValueError as e:
            st.error(e)
        try:
            importances = rfr.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#30B9EF')
        except Exception as e:
            st.error(e)
        st.session_state['model'] = rfr
        try:
            y_pred = rfr.predict(X_test)
        except Exception as e:
            st.error(e)
        st.session_state.predicted = y_pred
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        progress_bar = st.progress(0)
        status_text = st.empty()
        col6, col7 = st.columns([3,1])
        with col6.container(border=True):
            for i in range(1, 101):
                status_text.text("%i%% Complete" % i)
                progress_bar.progress(i)
                time.sleep(0.02)
            plot_chart(y_test, y_pred)
        with col7.container(border=True):
            acc = rfr.score(X_test, y_test)*100
            st.info(f'''**Error Margin**: {rmse.round(3)} \n\n **Accuracy**: {acc.round(2)}% \n\n **Time Took**: {round((time.time()-start),2)} secs''')
            
            
    else:
        rfc = RandomForestClassifier(max_depth = 5, max_leaf_nodes=12)
        st.session_state['model_type'] = 'classification'
        # steps = [('over', oversample)]
        # pipeline = Pipeline(steps=steps)
        # X_sm, y_sm = pipeline.fit_resample(X=X, y=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        
        try:
            rfc.fit(X_train, y_train)
        except Exception as e:
            st.error(e)
        try:
            importances = rfc.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#1168f5')
        except Exception as e:
            st.error(e)
        st.session_state['model'] = rfc
        try:
            y_pred = rfc.predict(X_test)
        except Exception as e:
            # st.warning("Can't predict due to some error in the Dependant/Independant Variable")
            st.error(e)
        st.session_state.predicted = y_pred
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(1, 101):
            status_text.text("%i%% Complete" % i)
            progress_bar.progress(i)
            time.sleep(0.02)
        acc = accuracy_score(y_test, y_pred)
        acc = round(acc, 2)
        cm = confusion_matrix(y_test, y_pred)
        col6, col7 = st.columns([3,1])
        with col7.container(border=True):
            st.info(f'''**Accuracy**: {round(acc, 2)*100}% \n\n **Time Took**: {round((time.time()-start),2)} secs''') 
        with col6.container(border=True):
            st.info("**Confusion Matrix**")
            fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=['True', 'False'],
                   y=['True', 'False'],
                   text=[[cm[0][0], cm[0][1]],
                         [cm[1][0], cm[1][1]]],
                   texttemplate='%{text}',
                   hoverongaps = True))
            fig.update_traces(showscale=False)
            st.plotly_chart(fig)

def plot_chart(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(50)], y=y_test[-50:], mode='markers', name='Testing points', line=dict(color="red")))
    fig.add_trace(go.Scatter(x=[i for i in range(50)], y=y_pred[-50:], mode='lines', name='Regression Line',opacity=0.9, line=dict(color="#30B9EF")))
    fig.update_layout(
    title="Regression Analysis",
    xaxis_title="Test Data",
    yaxis_title="Prediction Data",
    legend_title="Legend",
    )
    st.plotly_chart(fig, use_container_width=True)

# def oversampling(X_train, y_train):
#     smt=SMOTE()
#     X_train_sm, Y_train_sm = smt.fit_resample(X_train, y_train)
#     return X_train_sm, Y_train_sm