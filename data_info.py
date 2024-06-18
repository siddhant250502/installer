import streamlit as st
import pandas as pd
from functions import data_cleaning, plot_columns, correlation, r_square, interactive_plot
import plotly.express as px
from streamlit_plotly_events import plotly_events
from streamlit_option_menu import option_menu
import math
from streamlit_extras.stylable_container import stylable_container

def data_analysis():
    with st.container(border=True):
        option = option_menu(
        menu_title=None,
        options=['Data Header', 'Data Statistics', 'Correlation Matrix', 'Plot'],
        icons=['table', 'bar-chart-line', 'graph-up', 'diagram-2'],
        menu_icon=None,
        default_index=0,
        orientation='horizontal'
            )
        if option=='Data Header':
            prev_tab = st.session_state.prev_tab
            st.session_state.prev_tab = option
            if prev_tab=='Data Header':
                if st.session_state['data_header_df'] is not None:
                    df = st.session_state['data_header_df']
                else:
                    if st.session_state['file_path'] is not None:
                        df = pd.read_csv(st.session_state['file_path'])
                        df.columns = [i.upper() for i in df.columns]
                        df = data_cleaning(df)
                        st.session_state['filter_df'] = df
                        df['EXCLUDE/INCLUDE'] = True
                    else:
                        st.warning("File type Not supported")
                        
            elif prev_tab=='Data Statistics':
                if st.session_state['histogram_df'] is not None:
                    df = st.session_state['histogram_df']
                else:
                    df = st.session_state['data_header_df']
            elif prev_tab=='Correlation Matrix':
                if st.session_state['corr_df'] is not None:
                    df = st.session_state['corr_df']
                else:
                    df = st.session_state['histogram_df']
            elif prev_tab=='Plot':
                if st.session_state['filter_df'] is not None:
                    df = st.session_state['filter_df']
                else:
                    df = st.session_state['corr_df']
            
            with stylable_container(
                key='h3',
                css_styles="""
                    h3 {
                        font-size: 16px;
                    }
                """
            ):
                st.subheader('Header of the DataSet')
            c01, c02, c03 = st.columns(3)
            
            with c02:
                select = st.toggle('Select All')
            with c01:
                deselect = st.toggle('Deselect All')
            if select:
                df['EXCLUDE/INCLUDE'] = True
            if deselect:
                df['EXCLUDE/INCLUDE'] = False 
            try:
                with stylable_container(
                key='h3',
                css_styles="""
                    h3 {
                        font-size: 16px;
                    }
                """
                ):
                    st.subheader('Column Selection')
                res = []
                for i in df.columns:
                    res.append(True)
                res_df = pd.DataFrame(data=[res], columns=df.columns)
                res_df = st.data_editor(
                    res_df,
                    column_config={
                        f'{i}':st.column_config.CheckboxColumn()
                    },
                    # hide_index=True,
                    width=1600
                )
                cols = []
                for i in range(len(res_df.iloc[0])):
                    if res_df.iloc[0][i] == True:
                        cols.append(res_df.columns[i])
                df = df[cols]
                while True:
                    df = st.data_editor(
                        df,
                        column_config={
                            "Exclude/Include": st.column_config.CheckboxColumn()
                        },
                        use_container_width=True,
                        hide_index=False, 
                        height=350,
                        key='table'
                    )       
            except:
                pass
            if st.button("Save Data"):
                st.session_state['data_header_df'] = df[cols]
            
        elif option == 'Data Statistics':
            prev_tab = st.session_state.prev_tab
            st.session_state.prev_tab = option
            try:
                if prev_tab=='Data Header':
                    if st.session_state['data_header_df'] is None:
                        st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                        st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                        st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                        st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True  
                elif prev_tab=='Correlation Matrix':
                    if st.session_state['corr_df'] is not None:
                        st.session_state['data_header_df'] = st.session_state['corr_df']
                    else:
                        st.session_state['data_header_df'] = st.session_state['histogram_df']
                elif prev_tab=='Plot':
                    
                    if st.session_state['filter_df'] is not None:
                        st.session_state['data_header_df'] = st.session_state['filter_df']
                    else:
                        st.session_state['data_header_df'] = st.session_state['corr_df']
            
                with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                ):
                    st.subheader('Dataset statistics')
                with st.expander('Statistical info of the dataset'):
                    st.dataframe(st.session_state['data_header_df'].describe(), use_container_width=True)
                    st.write('----')
                with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                ):
                    st.subheader('Column Distribution ')
                opt = st.selectbox('Select any Column', options=st.session_state['data_header_df'].columns[:-1])
                c4, c10, c6, c100 = st.columns(4)
                with c4:
                    sigma = st.toggle('Real Stdev')
                with c10:
                    est_sigma = st.toggle('Est. Stdev')
                with c6:
                    curve = st.toggle('Curve')
                if opt:
                    bar_chart = plot_columns(st.session_state['data_header_df'], opt, sigma, est_sigma, curve)

                    bar_chart_selected = plotly_events(
                        bar_chart,
                        select_event=True,
                        
                    )
                new_df = pd.DataFrame(None, columns=[st.session_state['data_header_df'].columns])
                if bar_chart_selected:
                    x = (st.session_state['data_header_df'][opt].max()-st.session_state['data_header_df'][opt].min())/math.sqrt(len(st.session_state['data_header_df'][opt]))
                    for i in range(len(bar_chart_selected)):
                        min = bar_chart_selected[i]['x']-(x/2)
                        max = bar_chart_selected[i]['x']+(x/2)
                        temp_df = st.session_state['data_header_df'][st.session_state['data_header_df'][opt]>=min]
                        temp_df = temp_df[temp_df[opt]<=max]
                        new_df = pd.concat([temp_df, new_df])
                        new_df = new_df[new_df.columns[:len(st.session_state['data_header_df'].columns)]]
                        new_df = new_df.drop_duplicates()
                    with stylable_container(
                    key='h3',
                    css_styles="""
                        h3 {
                            font-size: 16px;
                        }
                    """
                    ):
                        st.subheader('Filtered Dataset Preview')   
                    st.dataframe(new_df[new_df['EXCLUDE/INCLUDE']==True], use_container_width=True, hide_index=True)     
                    st.session_state['histogram_df'] = new_df
                    st.session_state['histogram_df'].reset_index(inplace=True, drop=True)
            except AttributeError:
                st.error("You haven't saved the data in Data Header")
            except ValueError:
                st.error("You haven't saved the data in Data Header")
            
        elif option=='Correlation Matrix':
            prev_tab = st.session_state.prev_tab
            st.session_state.prev_tab = option
            # st.write(prev_tab)
            # tab1, tab2 = st.tabs(['Heatmap', 'Table'])
            if prev_tab=='Data Header':
                if st.session_state['data_header_df'] is not None:
                    # st.session_state['histogram_df'] = st.session_state['data_header_df']
                    heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                    st.session_state['corr_df'] = st.session_state['data_header_df']
                else:
                    st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                    st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                    st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                    st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True
                    heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                    st.session_state['corr_df'] = st.session_state['data_header_df']
            elif prev_tab=='Data Statistics':
                if st.session_state['histogram_df'] is not None:
                    # st.session_state['histogram_df'] = st.session_state['histogram_df']
                    heatmap = correlation(st.session_state['histogram_df'][st.session_state['histogram_df'].columns[:-1]])
                    st.session_state['corr_df'] = st.session_state['histogram_df']
                else:
                    # st.session_state['histogram_df'] = st.session_state['data_header_df']
                    if st.session_state['data_header_df'] is not None:
                        heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                        st.session_state['corr_df'] = st.session_state['data_header_df']
                    elif st.session_state['filter_df'] is not None:
                        heatmap = correlation(st.session_state['filter_df'][st.session_state['filter_df'].columns[:-1]])
                        st.session_state['corr_df'] = st.session_state['filter_df']
            elif prev_tab=='Plot':
                if st.session_state['filter_df'] is not None:
                    # st.session_state['histogram_df'] = st.session_state['filter_df']
                    heatmap = correlation(st.session_state['filter_df'][st.session_state['filter_df'].columns[:-1]])
                    st.session_state['corr_df'] = st.session_state['filter_df']
                else:
                    # st.session_state['histogram_df'] = st.session_state['corr_df']
                    heatmap = correlation(st.session_state['corr_df'][st.session_state['corr_df'].columns[:-1]])
                    st.session_state['corr_df'] = st.session_state['corr_df']
            
            if st.session_state['histogram_df'] is None:
                st.session_state['histogram_df'] = st.session_state['data_header_df']
            # with tab1:
            #     with stylable_container(
            #         key='h3',
            #         css_styles="""
            #             h3 {
            #                 font-size: 16px;
            #             }
            #         """
            #     ):
            #         st.subheader('Correlation Matrix Heatmap')
            #     heatmap = correlation(st.session_state['histogram_df'][st.session_state['histogram_df'].columns[:-1]])
            #     # config = {'displayModeBar':False}
            #     heatmap_selected = plotly_events(
            #         heatmap,
            #         click_event=True
            #     )
            #     # st.plotly_chart(heatmap, use_container_width=True, config=config)
                
            #     if heatmap_selected:
            #         st.session_state['col_name'].add(heatmap_selected[0]['x'])
            #         st.session_state['col_name'].add(heatmap_selected[0]['y'])
            #     cols = list(st.session_state['col_name'])
            #     if len(st.session_state['col_name'])!= 0:
            #         st.write("Columns selected:")
            #         st.text(st.session_state['col_name'])
            #     if st.button('Preview'):
            #         with stylable_container(
            #         key='h3',
            #         css_styles="""
            #             h3 {
            #                 font-size: 16px;
            #             }
            #         """
            #         ):
            #             st.subheader('Filtered Dataset Preview')
            #         st.session_state['corr_df'] = st.session_state['histogram_df'][cols]
            #         st.dataframe(st.session_state['corr_df'], hide_index=True, use_container_width=True)
            # with tab2:
            table = r_square(st.session_state['corr_df'][st.session_state['corr_df'].columns[:-1]])
            st.dataframe(table, width=600, hide_index=True)
                        
        elif option=='Plot':  
            prev_tab = st.session_state.prev_tab
            st.session_state.prev_tab = option
            # st.write(prev_tab)
            if prev_tab=='Data Header':
                if st.session_state['data_header_df'] is not None:
                    st.session_state['corr_df'] = st.session_state['data_header_df']
                else:
                    st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                    st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                    st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                    st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True
            elif prev_tab=='Data Statistics':
                if st.session_state['histogram_df'] is not None:
                    st.session_state['corr_df'] = st.session_state['histogram_df']
                else:
                    st.session_state['corr_df'] = st.session_state['data_header_df']
            elif prev_tab=='Correlation Matrix':
                if st.session_state['corr_df'] is not None:
                    st.session_state['corr_df'] = st.session_state['corr_df']
                else:
                    st.session_state['corr_df'] = st.session_state['histogram_df']
                
            if st.session_state['corr_df'] is None and st.session_state['histogram_df'] is not None:
                st.session_state['corr_df'] = st.session_state['histogram_df']
            elif st.session_state['corr_df'] is None and st.session_state['histogram_df'] is None:
                st.session_state['corr_df'] = st.session_state['data_header_df']
                
            st.session_state['EXCLUDE_df'] = st.session_state['corr_df']
            
            with stylable_container(
                key='h3',
                css_styles="""
                    h3 {
                        font-size: 16px;
                    }
                """
            ):
                st.subheader('Scatter Plot')
            x_axis = st.selectbox('X-axis', options=st.session_state['corr_df'].columns[:-1], index=len(st.session_state['corr_df'].columns)-3)
            y_axis = st.selectbox('Y-axis', options=st.session_state['corr_df'].columns[:-1], index=len(st.session_state['corr_df'].columns)-2)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ols = st.toggle('Regression line')
            # with c2:
            #     exchange = st.toggle('Show distribution')
            with c3:
                EXCLUDE = st.toggle('EXCLUDE Selected points')
            
            # OLS Results
            # df.drop(['Exluce/INCLUDE'], axis=1, inplace=True)
            fig = px.scatter(st.session_state['corr_df'], x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], trendline='ols') 
            res = px.get_trendline_results(fig)
            results = res.iloc[0]['px_fit_results']
            r2 = results.rsquared
            ttest = results.tvalues[0]
            ftest = results.f_pvalue
            line = y_axis+' = '+ str(results.params[1]) + ' * ' + x_axis +' + '+ str(results.params[0])
            pval = str(round(results.pvalues[1],3))
            res_df = pd.DataFrame.from_dict({"R-Squared":[r2], 'Line equation':[line], "P-Value":[pval], "f-test":[ftest], "T-Test":[ttest] })
            with stylable_container(
                key='h3',
                css_styles="""
                    h3 {
                        font-size: 16px;
                    }
                """
            ):
                st.subheader('OLS Results')
            st.dataframe(res_df, hide_index=True, use_container_width=True)
            # with stylable_container(
            #     key='tab',
            #     css_styles="""
            #         table {
            #             border: None !important;
            #             margin-top:20px;
            #             width:1400px;
            #         }
            #     """
            # ):    
            scatter_chart = interactive_plot(st.session_state['corr_df'], x_axis, y_axis, ols)
            with stylable_container(
                key='tab',
                css_styles="""
                    {
                        margin-top:30px;   
                    }
                """
            ):
                scatter_chart_selected = plotly_events(
                    scatter_chart,
                    select_event=True,
                )
            pntind = []
            if scatter_chart_selected:
                
                for i in range(len(scatter_chart_selected)):
                    pntind.append(st.session_state['corr_df'][(st.session_state['corr_df'][x_axis] == scatter_chart_selected[i]['x']) & (st.session_state['corr_df'][y_axis] == scatter_chart_selected[i]['y'])].index[0])
                if EXCLUDE:
                    st.session_state['EXCLUDE_df']['EXCLUDE/INCLUDE'].iloc[pntind] = False
                    st.session_state['filter_df'] = pd.concat([st.session_state['EXCLUDE_df'][st.session_state['EXCLUDE_df']['EXCLUDE/INCLUDE']==False], st.session_state['corr_df']])#[st.session_state['corr_df']['EXCLUDE/INCLUDE']==True]
                    st.session_state['filter_df'] = st.session_state['filter_df'].drop_duplicates(keep='first')
                    st.rerun()
                else:
                    st.session_state['filter_df'] = st.session_state['corr_df'].iloc[pntind]
                st.session_state['filter_df'].reset_index(inplace=True, drop=True)