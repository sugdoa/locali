import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Translation Quality Analysis Dashboard",
    page_icon="🌐",
    layout="wide"
)

# Title and description
st.title("Translation Quality Analysis Dashboard")
st.markdown("""
This dashboard helps analyze translation quality data and identify root causes of errors.
Upload your Excel file to get started.
""")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load the data
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File successfully loaded!")
        
        # Display raw data if requested
        if st.checkbox("Show raw data"):
            st.write(df)
        
        # Ensure necessary columns exist
        required_columns = [
            'TR Y', 'Error Points', 'WC', 'Issues', 'Issue Sub-Category', 
            'Issue Severity', 'TM Match Type', 'Issue Is Repeated', 'Date', 
            'Translator', 'Copy editor', 'Product', 'Content',
            'Arbitration Status', 'Linguistic', 'Meaning', 'Terminology', 
            'Fluency', 'Cultural', 'i18n', 'Consistency', 'Major',
            '100TM match', 'Repeated', 'Accepted', 'Rejected', 'Open',
            'Fuzzy', 'Machine translation', 'New', 'Owner/TR', 'RV/PR', 
            'Squad', 'Workflow'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Warning: The following required columns are missing: {', '.join(missing_columns)}")
        
        # Calculate WC2 if not present
        if 'WC2' not in df.columns:
            st.info("Calculating WC2 column...")
            # Assuming WC2's calculation logic - user needs to clarify this if different
            df['WC2'] = df['WC']
        
        # Calculate metrics based on formulas
        metrics = {}
        
        # Group metrics for each 1k words
        if 'WC2' in df.columns and df['WC2'].sum() > 0:
            metrics['Linguistic/1k'] = round((df['Linguistic'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['SS'] = round((100 - (df['Error Points'].sum() / df['WC2'].sum() * 1000)), 2)
            metrics['Meaning/1k'] = round((df['Meaning'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Terminology/1k'] = round((df['Terminology'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Fluency/1k'] = round((df['Fluency'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Cultural/1k'] = round((df['Cultural'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['i18n/1k'] = round((df['i18n'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Consistency/1k'] = round((df['Consistency'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Major/1k'] = round((df['Major'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['100TM match/1k'] = round((df['100TM match'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Repeated/1k'] = round((df['Repeated'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Accepted/1k'] = round((df['Accepted'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['Rejected/1k'] = round((df['Rejected'].sum() / df['WC2'].sum() * 1000), 2)
            
            # Calculate Accepted%
            if (df['Accepted'].sum() + df['Rejected'].sum() + df['Open'].sum()) > 0:
                metrics['Accepted%'] = round((df['Accepted'].sum() / (df['Accepted'].sum() + df['Rejected'].sum() + df['Open'].sum()) * 100), 2)
            else:
                metrics['Accepted%'] = 0
            
            # Continue with other metrics...
            metrics['Fuzzy/1k'] = round((df['Fuzzy'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['MT/1k'] = round((df['Machine translation'].sum() / df['WC2'].sum() * 1000), 2)
            metrics['New/1k'] = round((df['New'].sum() / df['WC2'].sum() * 1000), 2)
            
            # Need to check if these columns exist before calculating
            for column in ['Common Standards', 'Google linguistic reference materials', 'Google product/project instructions',
                          'Excellent', 'Creativity', 'Length & Segmentation', 'Timing', 'Positioning',
                          'Reading speed', 'Candidate for guidelines', 'Context missing', 'New instructions',
                          'Polishing', 'Source issue']:
                if column in df.columns:
                    metrics[f"{column}/1k"] = round((df[column].sum() / df['WC2'].sum() * 1000), 2)
            
            # Calculate Arb/1k
            if 'Open' in df.columns and 'Rejected' in df.columns and 'Accepted' in df.columns:
                metrics['Arb/1k'] = round(((df['Open'].sum() + df['Rejected'].sum() + df['Accepted'].sum()) / df['WC2'].sum() * 1000), 2)
            
            # Calculate Arbitrations/errors
            if 'Issues' in df.columns and df['Issues'].sum() > 0:
                metrics['Arbitrations/errors'] = round(((df['Accepted'].sum() + df['Rejected'].sum() + df['Open'].sum()) / df['Issues'].sum() * 100), 2)
                
        # Display calculated metrics
        st.subheader("Calculated Metrics")
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df)
        
        # Create Date columns if Date exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['week'] = df['Date'].dt.isocalendar().week
        
        # Analysis options
        st.subheader("Analysis Options")
        
        analysis_tab1, analysis_tab2 = st.tabs(["Service Score Analysis", "Issue Analysis"])
        
        with analysis_tab1:
            st.subheader("Service Score Analysis")
            
            # Create dimensions for analysis
            dimensions = ['Translator', 'Copy editor', 'Product', 'Content', 'month', 'week', 'Issue Category', 'Owner/TR', 'RV/PR', 'Squad', 'Workflow']
            available_dimensions = [dim for dim in dimensions if dim in df.columns]
            
            if available_dimensions:
                selected_dimension = st.selectbox("Select dimension for SS analysis:", available_dimensions)
                
                # Group by selected dimension and calculate SS
                grouped_df = df.groupby(selected_dimension).agg(
                    Total_WC=('WC2', 'sum'),
                    Total_Error_Points=('Error Points', 'sum')
                ).reset_index()
                
                # Calculate Service Score
                grouped_df['Service_Score'] = 100 - (grouped_df['Total_Error_Points'] / grouped_df['Total_WC'] * 1000)
                grouped_df['Service_Score'] = grouped_df['Service_Score'].round(2)
                
                # Sort by Service Score
                grouped_df = grouped_df.sort_values('Service_Score', ascending=False)
                
                # Display table
                st.write(grouped_df)
                
                # Create bar chart
                fig = px.bar(
                    grouped_df, 
                    x=selected_dimension, 
                    y='Service_Score',
                    title=f"Service Score by {selected_dimension}",
                    labels={'Service_Score': 'Service Score'},
                    color='Service_Score',
                    color_continuous_scale='RdYlGn',
                    text='Service_Score'
                )
                
                fig.update_layout(xaxis_title=selected_dimension, yaxis_title="Service Score")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No suitable dimensions found for analysis.")
        
        with analysis_tab2:
            st.subheader("Issue Analysis")
            
            # Create categories for issue analysis
            issue_categories = [
                'Linguistic', 'Meaning', 'Terminology', 'Fluency', 'Cultural', 
                'i18n', 'Consistency', 'Major', '100TM match', 'Repeated'
            ]
            
            available_issues = [issue for issue in issue_categories if issue in df.columns]
            
            if available_issues:
                # Create dimension selector for issues
                issue_dimension = st.selectbox("Select dimension for issue analysis:", available_dimensions)
                
                # Group by selected dimension and calculate issue metrics
                issue_df = df.groupby(issue_dimension).agg({
                    issue: 'sum' for issue in available_issues
                }).reset_index()
                
                issue_df['Total_WC'] = df.groupby(issue_dimension)['WC2'].sum().values
                
                # Calculate per 1k metrics
                for issue in available_issues:
                    issue_df[f"{issue}/1k"] = (issue_df[issue] / issue_df['Total_WC'] * 1000).round(2)
                
                # Display table
                st.write(issue_df)
                
                # Create visualization for issues
                selected_issue = st.selectbox("Select issue type to visualize:", available_issues)
                
                fig = px.bar(
                    issue_df, 
                    x=issue_dimension, 
                    y=f"{selected_issue}/1k",
                    title=f"{selected_issue} Issues per 1k Words by {issue_dimension}",
                    labels={f"{selected_issue}/1k": f"{selected_issue}/1k"},
                    color=f"{selected_issue}/1k",
                    color_continuous_scale='Reds',
                    text=f"{selected_issue}/1k"
                )
                
                fig.update_layout(xaxis_title=issue_dimension, yaxis_title=f"{selected_issue}/1k")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No issue categories found for analysis.")
                
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload an Excel file to begin analysis.")

# Add footer
st.markdown("---")
st.markdown("Translation Quality Analysis Dashboard - v1.0")

# Part 2: Advanced Analysis Features
# -------------------------------------------------------

# Create a new section for advanced analysis
st.header("Advanced Analysis", anchor="advanced-analysis")
st.markdown("""
This section provides more advanced analysis tools to help identify root causes of translation issues
and get deeper insights into your translation quality data.
""")

if uploaded_file is not None and 'df' in locals():
    # Multi-dimensional Analysis
    st.subheader("Multi-dimensional Analysis")
    
    # Create columns for filter selection
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Define potential filter dimensions
    filter_dimensions = [
        'Translator', 'Copy editor', 'Product', 'Content', 
        'Translation Type', 'Issue Category', 'Issue Sub-Category',
        'Issue Severity', 'TM Match Type', 'Owner/TR', 'RV/PR', 
        'Squad', 'Workflow'
    ]
    
    # Only show dimensions that exist in the dataframe
    available_filters = [dim for dim in filter_dimensions if dim in df.columns]
    
    # Add time dimensions if date exists
    if 'Date' in df.columns:
        if 'year' not in available_filters and 'year' in df.columns:
            available_filters.append('year')
        if 'month_name' not in available_filters and 'month_name' in df.columns:
            available_filters.append('month_name')
        if 'week' not in available_filters and 'week' in df.columns:
            available_filters.append('week')
    
    # Select filters
    with col1:
        filter1 = st.selectbox("1st Dimension:", 
                              ['None'] + available_filters, 
                              index=1 if len(available_filters) > 0 else 0)
    
    with col2:
        filter2 = st.selectbox("2nd Dimension:", 
                              ['None'] + available_filters,
                              index=2 if len(available_filters) > 1 else 0)
    
    with col3:
        filter3 = st.selectbox("3rd Dimension:", 
                              ['None'] + available_filters,
                              index=0)
    
    with col4:
        filter4 = st.selectbox("4th Dimension:", 
                              ['None'] + available_filters,
                              index=0)
    
    with col5:
        filter5 = st.selectbox("5th Dimension:", 
                              ['None'] + available_filters,
                              index=0)
    
    # Date range filter
    if 'Date' in df.columns:
        st.subheader("Date Range Filter")
        date_col1, date_col2 = st.columns(2)
        
        with date_col1:
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        
        with date_col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter dataframe by date
        filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    else:
        filtered_df = df.copy()
    
    # Apply multi-dimensional filters and analysis
    if filter1 != 'None' or filter2 != 'None' or filter3 != 'None' or filter4 != 'None' or filter5 != 'None':
        st.subheader("Multi-dimensional Analysis Results")
        
        # Create groupby dimensions list
        groupby_dims = []
        if filter1 != 'None':
            groupby_dims.append(filter1)
        if filter2 != 'None':
            groupby_dims.append(filter2)
        if filter3 != 'None':
            groupby_dims.append(filter3)
        if filter4 != 'None':
            groupby_dims.append(filter4)
        if filter5 != 'None':
            groupby_dims.append(filter5)
        
        if len(groupby_dims) > 0:
            # Group by the selected dimensions
            multi_grouped = filtered_df.groupby(groupby_dims).agg(
                Total_WC=('WC2', 'sum'),
                Total_Error_Points=('Error Points', 'sum'),
                Total_Issues=('Issues', 'sum')
            ).reset_index()
            
            # Calculate metrics
            multi_grouped['SS'] = (100 - (multi_grouped['Total_Error_Points'] / multi_grouped['Total_WC'] * 1000)).round(2)
            multi_grouped['Error_Rate'] = (multi_grouped['Total_Error_Points'] / multi_grouped['Total_WC'] * 1000).round(2)
            
            if len(multi_grouped) > 0:
                # Display the multi-dimensional table
                st.dataframe(
                    multi_grouped.style.background_gradient(
                        subset=['SS'], 
                        cmap='RdYlGn', 
                        vmin=0, 
                        vmax=100
                    ),
                    use_container_width=True
                )
                
                # Determine what type of visualization to show based on dimensions
                if len(groupby_dims) == 1:
                    # Simple bar chart for one dimension
                    fig = px.bar(
                        multi_grouped.sort_values('SS', ascending=False), 
                        x=groupby_dims[0], 
                        y='SS',
                        title=f"Service Score by {groupby_dims[0]}",
                        color='SS',
                        color_continuous_scale='RdYlGn',
                        text='SS',
                        hover_data=['Total_WC', 'Error_Rate']
                    )
                    fig.add_hline(y=90, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif len(groupby_dims) == 2:
                    # Heatmap for two dimensions
                    pivot_table = multi_grouped.pivot_table(
                        values='SS', 
                        index=groupby_dims[0], 
                        columns=groupby_dims[1]
                    )
                    
                    fig = px.imshow(
                        pivot_table,
                        labels=dict(x=groupby_dims[1], y=groupby_dims[0], color="SS"),
                        title=f"Service Score Heatmap: {groupby_dims[0]} vs {groupby_dims[1]}",
                        color_continuous_scale='RdYlGn',
                        range_color=[60, 100],
                        text_auto=True
                    )
                    
                    fig.update_layout(
                        xaxis_title=groupby_dims[1],
                        yaxis_title=groupby_dims[0]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif len(groupby_dims) >= 3:
                    # For 3+ dimensions, use enhanced bubble chart
                    st.markdown(f"### Multi-dimensional Analysis ({len(groupby_dims)} dimensions)")
                    st.markdown(f"Primary dimensions: {', '.join(groupby_dims[:3])}")
                    if len(groupby_dims) > 3:
                        st.markdown(f"Additional dimensions: {', '.join(groupby_dims[3:])}")
                    
                    # Let user choose which dimension to use for coloring
                    color_dim = st.selectbox(
                        "Select dimension for color coding:",
                        groupby_dims,
                        index=0
                    )
                    
                    # Let user choose X and Y axes
                    x_dim = st.selectbox(
                        "Select dimension for X axis:",
                        groupby_dims + ['Total_WC'],
                        index=1 if len(groupby_dims) > 1 else 0
                    )
                    
                    # Create scatter plot with bubble size based on word count
                    hover_data = ['Error_Rate', 'Total_Issues']
                    for dim in groupby_dims:
                        if dim != color_dim and dim != x_dim:
                            hover_data.append(dim)
                    
                    fig = px.scatter(
                        multi_grouped,
                        x=x_dim,
                        y='SS',
                        color=color_dim,
                        size='Total_WC',
                        hover_data=hover_data,
                        title=f"Service Score Analysis by Multiple Dimensions",
                        labels={'SS': 'Service Score'}
                    )
                    
                    fig.add_hline(y=90, line_dash="dash", line_color="red")
                    
                    fig.update_layout(
                        xaxis_title=x_dim,
                        yaxis_title="Service Score",
                        yaxis=dict(range=[60, 105])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")
    
    # Root Cause Analysis
    st.header("Root Cause Analysis")
    st.markdown("""
    This section helps identify root causes of translation issues by analyzing correlations
    between different factors and error rates.
    """)
    
    # Create columns for analysis options
    root_col1, root_col2 = st.columns(2)
    
    with root_col1:
        # Choose error category to analyze
        error_categories = [
            'Linguistic', 'Meaning', 'Terminology', 'Fluency', 'Cultural', 
            'i18n', 'Consistency'
        ]
        
        available_errors = [err for err in error_categories if err in filtered_df.columns]
        
        if available_errors:
            selected_error = st.selectbox(
                "Select error category to analyze:",
                ['Error Points'] + available_errors,
                index=0
            )
        else:
            selected_error = 'Error Points'
            st.warning("No specific error categories found. Using total Error Points.")
    
    with root_col2:
        # Choose dimension to analyze against
        if filter1 != 'None':
            analysis_dimension = st.selectbox(
                "Analyze by dimension:",
                available_filters,
                index=available_filters.index(filter1) if filter1 in available_filters else 0
            )
        else:
            analysis_dimension = st.selectbox(
                "Analyze by dimension:",
                available_filters,
                index=0 if len(available_filters) > 0 else 0
            )
    
    if selected_error and analysis_dimension and analysis_dimension in filtered_df.columns:
        # Group data for root cause analysis
        if selected_error == 'Error Points':
            error_field = 'Error Points'
        else:
            error_field = selected_error
        
        # Verify the error field exists in the dataframe
        if error_field in filtered_df.columns:
            # Group by the analysis dimension
            root_cause_df = filtered_df.groupby(analysis_dimension).agg(
                Total_WC=('WC2', 'sum'),
                Total_Errors=(error_field, 'sum')
            ).reset_index()
            
            # Calculate error rate per 1k words
            root_cause_df['Error_Rate'] = (root_cause_df['Total_Errors'] / root_cause_df['Total_WC'] * 1000).round(2)
            
            # Sort by error rate descending
            root_cause_df = root_cause_df.sort_values('Error_Rate', ascending=False)
            
            # Display top contributors to errors
            st.subheader(f"Top Contributors to {selected_error}")
            st.dataframe(root_cause_df, use_container_width=True)
            
            # Create visualization of top contributors
            fig = px.bar(
                root_cause_df.head(10),
                x=analysis_dimension,
                y='Error_Rate',
                title=f"Top Contributors to {selected_error} (per 1k words)",
                color='Error_Rate',
                color_continuous_scale='Reds',
                text='Error_Rate',
                hover_data=['Total_WC', 'Total_Errors']
            )
            
            fig.update_layout(
                xaxis_title=analysis_dimension,
                yaxis_title=f"{selected_error} Rate per 1k words"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pareto Analysis - 80/20 rule
            st.subheader("Pareto Analysis (80/20 Rule)")
            
            # Sort by total errors
            pareto_df = root_cause_df.sort_values('Total_Errors', ascending=False).copy()
            
            # Calculate cumulative errors and percentages
            pareto_df['Cumulative_Errors'] = pareto_df['Total_Errors'].cumsum()
            total_errors = pareto_df['Total_Errors'].sum()
            pareto_df['Error_Percentage'] = (pareto_df['Total_Errors'] / total_errors * 100).round(1)
            pareto_df['Cumulative_Percentage'] = (pareto_df['Cumulative_Errors'] / total_errors * 100).round(1)
            
            # Display Pareto table
            st.dataframe(pareto_df[
                [analysis_dimension, 'Total_Errors', 'Error_Percentage', 'Cumulative_Percentage']
            ], use_container_width=True)
            
            # Create Pareto chart
            fig = go.Figure()
            
            # Add bar chart
            fig.add_trace(go.Bar(
                x=pareto_df[analysis_dimension],
                y=pareto_df['Error_Percentage'],
                name='Error Percentage',
                marker_color='indianred'
            ))
            
            # Add line chart for cumulative percentage
            fig.add_trace(go.Scatter(
                x=pareto_df[analysis_dimension],
                y=pareto_df['Cumulative_Percentage'],
                name='Cumulative Percentage',
                marker_color='royalblue',
                mode='lines+markers'
            ))
            
            # Add 80% threshold line
            fig.add_hline(
                y=80, 
                line_dash="dash", 
                line_color="green",
                annotation_text="80% Threshold",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Pareto Analysis for {selected_error}",
                xaxis_title=analysis_dimension,
                yaxis_title="Percentage (%)",
                yaxis=dict(range=[0, 105]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find the critical few contributors that cause 80% of errors
            critical_contributors = pareto_df[pareto_df['Cumulative_Percentage'] <= 80]
            
            if not critical_contributors.empty:
                st.markdown(f"### Critical Contributors (80% of {selected_error})")
                st.markdown(f"**{len(critical_contributors)}** out of **{len(pareto_df)}** {analysis_dimension}s contribute to **80%** of all {selected_error}.")
                st.dataframe(critical_contributors[[analysis_dimension, 'Error_Percentage', 'Cumulative_Percentage']])
            
        else:
            st.warning(f"Error field '{error_field}' not found in the data.")
    
    # Automated Insights
    st.header("Automated Insights")
    
    if 'SS' in metrics:
        overall_ss = metrics['SS']
        
        insights = []
        
        # Add overall service score insight
        if overall_ss >= 95:
            insights.append(f"🟢 Overall Service Score is excellent at {overall_ss}%.")
        elif overall_ss >= 90:
            insights.append(f"🟡 Overall Service Score is good at {overall_ss}%, but there's room for improvement.")
        else:
            insights.append(f"🔴 Overall Service Score needs improvement at {overall_ss}%.")
        
        # Time trend insights if date exists
        if 'Date' in filtered_df.columns and len(filtered_df) > 1:
            # Group by month for trend analysis
            if 'month' in filtered_df.columns and 'year' in filtered_df.columns:
                filtered_df['year_month'] = filtered_df['year'].astype(str) + '-' + filtered_df['month'].astype(str).str.zfill(2)
                trend_df = filtered_df.groupby('year_month').agg(
                    Total_WC=('WC2', 'sum'),
                    Total_Error_Points=('Error Points', 'sum')
                ).reset_index()
                
                # Calculate SS for each month
                trend_df['Monthly_SS'] = (100 - (trend_df['Total_Error_Points'] / trend_df['Total_WC'] * 1000)).round(2)
                
                # Sort by year_month
                trend_df = trend_df.sort_values('year_month')
                
                # Check for trends
                if len(trend_df) >= 3:
                    last_3_months = trend_df.tail(3)
                    first_ss = last_3_months.iloc[0]['Monthly_SS']
                    middle_ss = last_3_months.iloc[1]['Monthly_SS']
                    last_ss = last_3_months.iloc[2]['Monthly_SS']
                    
                    if last_ss > middle_ss > first_ss:
                        insights.append(f"📈 Positive trend detected: Service Score has been improving over the last 3 months ({first_ss}% → {middle_ss}% → {last_ss}%).")
                    elif last_ss < middle_ss < first_ss:
                        insights.append(f"📉 Negative trend detected: Service Score has been declining over the last 3 months ({first_ss}% → {middle_ss}% → {last_ss}%).")
        
        # Identify top errorop error categories
        error_categories = [
            'Linguistic', 'Meaning', 'Terminology', 'Fluency', 'Cultural', 
            'i18n', 'Consistency'
        ]
        
        available_errors = [err for err in error_categories if err in filtered_df.columns]
        
        if available_errors:
            error_totals = {}
            for error in available_errors:
                if 'WC2' in filtered_df.columns and filtered_df['WC2'].sum() > 0:
                    error_rate = (filtered_df[error].sum() / filtered_df['WC2'].sum() * 1000).round(2)
                    error_totals[error] = error_rate
            
            # Sort and get top 3 error categories
            sorted_errors = sorted(error_totals.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_errors:
                top_error = sorted_errors[0]
                insights.append(f"🔍 The most common error category is '{top_error[0]}' with {top_error[1]} errors per 1k words.")
                
                if len(sorted_errors) >= 3:
                    insights.append(f"💡 Top 3 error categories to focus on: {sorted_errors[0][0]} ({sorted_errors[0][1]}), {sorted_errors[1][0]} ({sorted_errors[1][1]}), and {sorted_errors[2][0]} ({sorted_errors[2][1]}) per 1k words.")
        
        # Add insights about new columns (Owner/TR, RV/PR, Squad, Workflow)
        new_columns = ['Owner/TR', 'RV/PR', 'Squad', 'Workflow']
        for col in new_columns:
            if col in filtered_df.columns:
                try:
                    # Get top performer for each column
                    top_df = filtered_df.groupby(col).agg(
                        Total_WC=('WC2', 'sum'),
                        Total_Error_Points=('Error Points', 'sum')
                    ).reset_index()
                    
                    # Calculate SS
                    top_df['SS'] = (100 - (top_df['Total_Error_Points'] / top_df['Total_WC'] * 1000)).round(2)
                    
                    # Filter for meaningful data (at least 1000 words)
                    top_df = top_df[top_df['Total_WC'] >= 1000]
                    
                    if not top_df.empty:
                        # Get the best and worst performers
                        best = top_df.loc[top_df['SS'].idxmax()]
                        worst = top_df.loc[top_df['SS'].idxmin()]
                        
                        # Add insights
                        if best['SS'] > 90:
                            insights.append(f"⭐ Best {col}: '{best[col]}' with SS of {best['SS']}% ({best['Total_WC']} words)")
                        
                        if worst['SS'] < 85 and best[col] != worst[col]:
                            insights.append(f"⚠️ {col} requiring attention: '{worst[col]}' with SS of {worst['SS']}% ({worst['Total_WC']} words)")
                except:
                    pass
                
        # Display all insights
        for i, insight in enumerate(insights):
            st.markdown(f"{i+1}. {insight}")
    
    # Export analysis
    st.header("Export Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export filtered data
        if st.button("Export Filtered Data to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_translation_data.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export metrics
        if 'metrics_df' in locals():
            if st.button("Export Calculated Metrics to CSV"):
                metrics_csv = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Download Metrics CSV",
                    data=metrics_csv,
                    file_name="translation_metrics.csv",
                    mime="text/csv"
                )
    
    # Custom Thresholds
    st.header("Custom Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ss_threshold = st.slider(
            "Service Score Threshold (%)",
            min_value=0,
            max_value=100,
            value=90,
            step=1,
            help="Set custom threshold for Service Score"
        )
    
    with col2:
        error_threshold = st.slider(
            "Error Rate Threshold (per 1k words)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Set custom threshold for Error Rate"
        )
    
    # Apply thresholds to show potential issues
    if 'multi_grouped' in locals() and len(multi_grouped) > 0:
        st.subheader("Entries Below Service Score Threshold")
        
        # Filter by threshold
        below_threshold = multi_grouped[multi_grouped['SS'] < ss_threshold]
        
        if not below_threshold.empty:
            st.dataframe(
                below_threshold.sort_values('SS').style.background_gradient(
                    subset=['SS'], 
                    cmap='RdYlGn', 
                    vmin=0, 
                    vmax=100
                ),
                use_container_width=True
            )
            
            # Create visualization
            if len(groupby_dims) == 1:
                fig = px.bar(
                    below_threshold.sort_values('SS'),
                    x=groupby_dims[0],
                    y='SS',
                    title=f"Entries Below Service Score Threshold ({ss_threshold}%)",
                    color='SS',
                    color_continuous_scale='RdYlGn',
                    text='SS',
                    hover_data=['Total_WC', 'Error_Rate']
                )
                
                fig.add_hline(y=ss_threshold, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    xaxis_title=groupby_dims[0],
                    yaxis_title="Service Score",
                    yaxis=dict(range=[min(below_threshold['SS'].min() - 5, 60), ss_threshold + 5])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success(f"All entries are above the Service Score threshold of {ss_threshold}%!")
    
    # Advanced Visualizations
    st.header("Advanced Visualizations")
    
    # Translator vs Copy Editor Performance
    if 'Translator' in filtered_df.columns and 'Copy editor' in filtered_df.columns:
        st.subheader("Translator vs Copy Editor Performance")
        
        # Group by translator and copy editor
        translator_ce_df = filtered_df.groupby(['Translator', 'Copy editor']).agg(
            Total_WC=('WC2', 'sum'),
            Total_Error_Points=('Error Points', 'sum')
        ).reset_index()
        
        # Calculate SS
        translator_ce_df['SS'] = (100 - (translator_ce_df['Total_Error_Points'] / translator_ce_df['Total_WC'] * 1000)).round(2)
        
        # Filter out combinations with low word count
        min_wc_threshold = st.slider(
            "Minimum Word Count Threshold",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        translator_ce_df = translator_ce_df[translator_ce_df['Total_WC'] >= min_wc_threshold]
        
        if not translator_ce_df.empty:
            # Create heatmap
            pivot_table = translator_ce_df.pivot_table(
                values='SS', 
                index='Translator', 
                columns='Copy editor'
            )
            
            fig = px.imshow(
                pivot_table,
                labels=dict(x="Copy Editor", y="Translator", color="Service Score"),
                title="Service Score by Translator and Copy Editor",
                color_continuous_scale='RdYlGn',
                range_color=[60, 100],
                text_auto=True
            )
            
            fig.update_layout(
                xaxis_title="Copy Editor",
                yaxis_title="Translator"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add bubble chart with Total_WC as size
            fig = px.scatter(
                translator_ce_df,
                x='Translator',
                y='Copy editor',
                size='Total_WC',
                color='SS',
                color_continuous_scale='RdYlGn',
                range_color=[60, 100],
                hover_data=['SS', 'Total_Error_Points'],
                title="Service Score by Translator and Copy Editor (Bubble Size = Word Count)"
            )
            
            fig.update_layout(
                xaxis_title="Translator",
                yaxis_title="Copy Editor"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available with word count >= {min_wc_threshold}")
    
    # Analysis of new columns (Owner/TR, RV/PR, Squad, Workflow)
    new_columns_available = [col for col in ['Owner/TR', 'RV/PR', 'Squad', 'Workflow'] if col in filtered_df.columns]
    
    if new_columns_available:
        st.subheader("New Dimensions Analysis")
        
        selected_new_dim = st.selectbox(
            "Select new dimension to analyze:",
            new_columns_available
        )
        
        # Analyze the selected dimension
        if selected_new_dim in filtered_df.columns:
            new_dim_df = filtered_df.groupby(selected_new_dim).agg(
                Total_WC=('WC2', 'sum'),
                Total_Error_Points=('Error Points', 'sum')
            ).reset_index()
            
            # Calculate SS
            new_dim_df['SS'] = (100 - (new_dim_df['Total_Error_Points'] / new_dim_df['Total_WC'] * 1000)).round(2)
            
            # Filter for meaningful data (at least 100 words)
            new_dim_df = new_dim_df[new_dim_df['Total_WC'] >= 100]
            
            if not new_dim_df.empty:
                # Sort by SS
                new_dim_df = new_dim_df.sort_values('SS', ascending=False)
                
                # Display table
                st.dataframe(new_dim_df, use_container_width=True)
                
                # Create visualization
                fig = px.bar(
                    new_dim_df,
                    x=selected_new_dim,
                    y='SS',
                    color='SS',
                    color_continuous_scale='RdYlGn',
                    range_color=[60, 100],
                    text='SS',
                    hover_data=['Total_WC', 'Total_Error_Points'],
                    title=f"Service Score by {selected_new_dim}"
                )
                
                fig.add_hline(y=90, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    xaxis_title=selected_new_dim,
                    yaxis_title="Service Score"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create sunburst chart with one of the original dimensions
                if filter1 != 'None' and filter1 != selected_new_dim and filter1 in filtered_df.columns:
                    # Create a sunburst visualization
                    st.subheader(f"Sunburst Analysis: {selected_new_dim} and {filter1}")
                    
                    sunburst_df = filtered_df.groupby([selected_new_dim, filter1]).agg(
                        Total_WC=('WC2', 'sum'),
                        Total_Error_Points=('Error Points', 'sum')
                    ).reset_index()
                    
                    # Calculate SS
                    sunburst_df['SS'] = (100 - (sunburst_df['Total_Error_Points'] / sunburst_df['Total_WC'] * 1000)).round(2)
                    
                    # Filter for meaningful data
                    sunburst_df = sunburst_df[sunburst_df['Total_WC'] >= 100]
                    
                    if not sunburst_df.empty:
                        fig = px.sunburst(
                            sunburst_df, 
                            path=[selected_new_dim, filter1], 
                            values='Total_WC',
                            color='SS',
                            color_continuous_scale='RdYlGn',
                            range_color=[60, 100],
                            hover_data=['Total_Error_Points'],
                            title=f"Service Score by {selected_new_dim} and {filter1}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Word Count vs Error Rate Analysis
    st.subheader("Word Count vs Error Rate Analysis")
    
    if filter1 != 'None' and filter1 in filtered_df.columns:
        # Group by the selected dimension and calculate metrics
        scatter_df = filtered_df.groupby(filter1).agg(
            Total_WC=('WC2', 'sum'),
            Total_Error_Points=('Error Points', 'sum')
        ).reset_index()
        
        # Calculate Error Rate
        scatter_df['Error_Rate'] = (scatter_df['Total_Error_Points'] / scatter_df['Total_WC'] * 1000).round(2)
        
        # Create scatter plot
        fig = px.scatter(
            scatter_df,
            x='Total_WC',
            y='Error_Rate',
            color='Error_Rate',
            color_continuous_scale='Reds',
            hover_name=filter1,
            size='Total_WC',
            title=f"Word Count vs Error Rate by {filter1}",
            labels={
                'Total_WC': 'Total Word Count',
                'Error_Rate': 'Error Rate (per 1k words)'
            }
        )
        
        fig.add_hline(y=error_threshold, line_dash="dash", line_color="red")
        
        fig.update_layout(
            xaxis_title="Total Word Count",
            yaxis_title="Error Rate (per 1k words)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        corr = np.corrcoef(scatter_df['Total_WC'], scatter_df['Error_Rate'])[0, 1]
        
        st.markdown(f"Correlation between Word Count and Error Rate: **{corr:.2f}**")
        
        if abs(corr) > 0.5:
            if corr > 0:
                st.markdown("📈 There seems to be a **positive correlation** between word count and error rate. Higher word counts tend to have higher error rates.")
            else:
                st.markdown("📉 There seems to be a **negative correlation** between word count and error rate. Higher word counts tend to have lower error rates.")
        else:
            st.markdown("🔄 There is no strong correlation between word count and error rate.")
    
    # Five-dimensional combined analysis
    if all(x != 'None' for x in [filter1, filter2, filter3, filter4, filter5]):
        st.header("Five-Dimensional Combined Analysis")
        
        st.markdown("""
        This special analysis combines all five selected dimensions to identify patterns and outliers
        in your translation quality data. Each dimension contributes to the overall analysis.
        """)
        
        # Create a unique identifier for each combination
        five_dim_df = filtered_df.copy()
        five_dim_df['dimension_key'] = ''
        for dim in [filter1, filter2, filter3, filter4, filter5]:
            five_dim_df['dimension_key'] += five_dim_df[dim].astype(str) + " | "
        
        # Group by the combined key
        combined_df = five_dim_df.groupby('dimension_key').agg(
            Total_WC=('WC2', 'sum'),
            Total_Error_Points=('Error Points', 'sum')
        ).reset_index()
        
        # Calculate SS
        combined_df['SS'] = (100 - (combined_df['Total_Error_Points'] / combined_df['Total_WC'] * 1000)).round(2)
        
        # Filter for meaningful data
        combined_df = combined_df[combined_df['Total_WC'] >= 500]
        
        if not combined_df.empty:
            # Sort by SS
            combined_df = combined_df.sort_values('SS', ascending=False)
            
            # Extract the key back into separate columns for better display
            for i, dim in enumerate([filter1, filter2, filter3, filter4, filter5]):
                combined_df[dim] = combined_df['dimension_key'].str.split(' | ', expand=True)[i]
            
            # Display table with the individual dimensions and metrics
            st.dataframe(
                combined_df[[filter1, filter2, filter3, filter4, filter5, 'Total_WC', 'SS']].style.background_gradient(
                    subset=['SS'], 
                    cmap='RdYlGn', 
                    vmin=0, 
                    vmax=100
                ),
                use_container_width=True
            )
            
            # Create a parallel coordinates plot
            fig = px.parallel_coordinates(
                combined_df, 
                dimensions=[filter1, filter2, filter3, filter4, filter5, 'SS'],
                color='SS',
                color_continuous_scale='RdYlGn',
                range_color=[60, 100],
                title="Five-Dimensional Parallel Coordinates Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a 3D scatter plot using three of the dimensions
            st.subheader("3D Visualization of Selected Dimensions")
            
            # Select which dimensions to use for 3D plot
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_dim = st.selectbox("X-axis dimension:", [filter1, filter2, filter3, filter4, filter5], index=0)
            
            with col2:
                y_dim = st.selectbox("Y-axis dimension:", [filter1, filter2, filter3, filter4, filter5], index=1)
            
            with col3:
                z_dim = st.selectbox("Z-axis dimension:", [filter1, filter2, filter3, filter4, filter5], index=2)
            
            if x_dim != y_dim and x_dim != z_dim and y_dim != z_dim:
                # Create 3D scatter plot
                fig = px.scatter_3d(
                    combined_df,
                    x=x_dim,
                    y=y_dim,
                    z=z_dim,
                    color='SS',
                    size='Total_WC',
                    hover_data=[filter1, filter2, filter3, filter4, filter5, 'SS'],
                    color_continuous_scale='RdYlGn',
                    range_color=[60, 100],
                    title="3D Visualization of Selected Dimensions"
                )
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_dim,
                        yaxis_title=y_dim,
                        zaxis_title=z_dim
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload an Excel file in the section above to enable advanced analysis.")

# Add footer for Part 2
st.markdown("---")
st.markdown("Translation Quality Analysis Dashboard - Advanced Analysis v1.0")
