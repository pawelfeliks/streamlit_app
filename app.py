import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from io import BytesIO
from reportlab.pdfgen import canvas
import base64
import threading
import time
from datetime import datetime

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandasql as ps  # For SQL-like queries on DataFrame

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Data Upload'

if 'dummy_var' not in st.session_state:
    st.session_state.dummy_var = 0  # Dummy variable to trigger rerun

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Define page functions mapping
page_functions = {
    'Data Upload': lambda: page1_data_upload(),
    'Data Cleaning': lambda: page2_data_cleaning(),
    'Data Analysis': lambda: page3_data_analysis(),
    'Visualization': lambda: page4_visualization(),
    'Predict Trends': lambda: page5_predict_trends(),
    'Clustering': lambda: page6_clustering(),
    'Custom Dashboard': lambda: page7_custom_dashboard(),
    'NLP Query': lambda: page8_nlp_query(),
    'Real-Time Data': lambda: page9_real_time_data(),
    'Export Data': lambda: page10_export_data(),
    'Help': lambda: page11_help()
}

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", list(page_functions.keys()), index=list(page_functions.keys()).index(st.session_state.current_page))

# Update current page based on sidebar selection
st.session_state.current_page = options

# Function to load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.parquet'):
        return pd.read_parquet(file)
    else:
        st.error("Unsupported file format!")
        return pd.DataFrame()

# Data Sampling Function
def sample_data(df, frac=0.1):
    return df.sample(frac=frac)

# Define functions for each page
def page1_data_upload():
    st.title("Step 1: Data Upload")
    st.write("Supported file formats: CSV, Excel, Parquet")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "parquet"],
        help="Upload a CSV, Excel, or Parquet file for analysis."
    )

    # Data Sampling Option
    sample_option = st.checkbox("Sample Data for Large Files", value=False)
    sample_fraction = st.slider(
        "Select Sample Fraction",
        0.01, 1.0, 0.1, 0.01
    ) if sample_option else 1.0

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            if sample_option and len(df) > 10000:
                df = sample_data(df, frac=sample_fraction)
                st.warning(
                    f"Data has been sampled to {sample_fraction*100}% "
                    "of the original size for performance."
                )
            st.success("File uploaded successfully!")
            st.session_state.df = df
            st.write(df.head())
            if st.button("Next", key=f"next_{st.session_state.current_page}"):
                st.session_state.current_page = 'Data Cleaning'
                st.session_state.dummy_var += 1
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
    else:
        st.info("Please upload a dataset to proceed.")

def page2_data_cleaning():
    st.title("Step 2: Data Cleaning")
    st.write("Identify and handle missing values.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload data in the 'Data Upload' section before proceeding.")
        return

    # Show missing values
    if st.checkbox(
        "Show missing values summary",
        help="Check this box to see the summary of missing values in your dataset."
    ):
        st.write(df.isna().sum())

    # Fill missing values
    fill_method = st.selectbox(
        "Select method to fill missing values",
        ['Mean', 'Median', 'Mode', 'Constant Value']
    )
    if fill_method == 'Constant Value':
        constant_value = st.number_input(
            "Enter the constant value to fill missing values",
            value=0
        )
    if st.button("Fill missing values"):
        if fill_method == 'Mean':
            df.fillna(df.mean(), inplace=True)
        elif fill_method == 'Median':
            df.fillna(df.median(), inplace=True)
        elif fill_method == 'Mode':
            df.fillna(df.mode().iloc[0], inplace=True)
        elif fill_method == 'Constant Value':
            df.fillna(constant_value, inplace=True)
        st.success(f"Missing values filled using {fill_method} method!")
        st.session_state.df = df

    # Data type conversion
    st.subheader("Data Type Conversion")
    columns = df.columns.tolist()
    if columns:
        column_to_convert = st.selectbox(
            "Select column to convert data type",
            columns
        )
        new_data_type = st.selectbox(
            "Select new data type",
            ['int', 'float', 'str']
        )
        if st.button("Convert Data Type"):
            try:
                df[column_to_convert] = df[column_to_convert].astype(new_data_type)
                st.success(
                    f"Column '{column_to_convert}' converted to {new_data_type}!"
                )
                st.session_state.df = df
            except Exception as e:
                st.error(f"An error occurred during data type conversion: {e}")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Data Analysis'
        st.session_state.dummy_var += 1

def page3_data_analysis():
    st.title("Step 3: Data Analysis")
    st.write("Filter, sort, and summarize your data.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and clean data before proceeding.")
        return

    # Interactive Filtering
    st.subheader("Filter Data")
    columns = df.columns.tolist()
    if columns:
        selected_column = st.selectbox(
            "Select column to filter by",
            columns,
            help="Choose a column to filter the data."
        )
        unique_values = df[selected_column].dropna().unique()
        selected_values = st.multiselect(
            "Select values",
            unique_values,
            help="Select one or more values to filter the data."
        )
        if selected_values:
            filtered_df = df[df[selected_column].isin(selected_values)]
        else:
            filtered_df = df.copy()
        st.session_state.filtered_df = filtered_df
    else:
        st.error("No columns available in the dataset.")

    # Numeric Filtering with Slider
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_columns:
        st.subheader("Numeric Filtering")
        selected_numeric_column = st.selectbox(
            "Select a numeric column to filter",
            numeric_columns,
            help="Choose a numeric column for further filtering."
        )
        min_val = float(df[selected_numeric_column].min())
        max_val = float(df[selected_numeric_column].max())
        range_values = st.slider(
            f"Select range for {selected_numeric_column}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
        filtered_df = filtered_df[
            (filtered_df[selected_numeric_column] >= range_values[0]) &
            (filtered_df[selected_numeric_column] <= range_values[1])
        ]
        st.session_state.filtered_df = filtered_df

    # Data Sorting
    st.subheader("Sort Data")
    sort_column = st.selectbox(
        "Select column to sort by",
        columns
    )
    sort_order = st.radio("Select sort order", ("Ascending", "Descending"))
    filtered_df = filtered_df.sort_values(
        by=sort_column,
        ascending=(sort_order == "Ascending")
    )
    st.session_state.filtered_df = filtered_df
    st.write(filtered_df)

    # Data Summary
    st.subheader("Data Summary")
    if st.button(
        "Show Summary",
        key='show_summary',
        help="Click to see statistical summary of the filtered data."
    ):
        st.write(filtered_df.describe())

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Visualization'
        st.session_state.dummy_var += 1

def page4_visualization():
    st.title("Step 4: Data Visualization")
    st.write("Create interactive charts to visualize your data.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    # Select data source
    data_source = st.selectbox(
        "Select data source for plotting",
        ["Original Data", "Filtered Data"],
        help="Choose which data to visualize."
    )
    st.session_state['data_source'] = data_source

    # Select chart type
    chart_types = [
        "Line Chart", "Bar Chart", "Scatter Plot", "Histogram",
        "Box Plot", "Area Chart", "Pie Chart", "Heatmap", "Pairplot"
    ]
    chart_type = st.selectbox(
        "Select chart type",
        chart_types,
        help="Choose the type of chart to create."
    )
    st.session_state['chart_type'] = chart_type

    # Get data based on selection
    if data_source == "Original Data":
        plot_data = df.copy()
    else:
        plot_data = st.session_state.get('filtered_df', df)

    # Select columns for plotting
    columns_plot = plot_data.columns.tolist()
    if columns_plot:
        x_column = st.selectbox("Select x-axis column", columns_plot)
        y_column = st.selectbox("Select y-axis column", columns_plot)
        st.session_state['x_column'] = x_column
        st.session_state['y_column'] = y_column

        # Generate Plot
        if st.button("Generate Plot"):
            try:
                if chart_type == "Line Chart":
                    fig = px.line(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} vs {x_column}"
                    )
                elif chart_type == "Bar Chart":
                    fig = px.bar(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} vs {x_column}"
                    )
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} vs {x_column}"
                    )
                elif chart_type == "Histogram":
                    fig = px.histogram(
                        plot_data,
                        x=y_column,
                        nbins=20,
                        title=f"{chart_type} of {y_column}"
                    )
                elif chart_type == "Box Plot":
                    fig = px.box(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} by {x_column}"
                    )
                elif chart_type == "Area Chart":
                    fig = px.area(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} vs {x_column}"
                    )
                elif chart_type == "Pie Chart":
                    fig = px.pie(
                        plot_data,
                        names=x_column,
                        values=y_column,
                        title=f"{chart_type} of {y_column} by {x_column}"
                    )
                elif chart_type == "Heatmap":
                    fig = px.density_heatmap(
                        plot_data,
                        x=x_column,
                        y=y_column,
                        title=f"{chart_type} of {y_column} vs {x_column}"
                    )
                elif chart_type == "Pairplot":
                    fig = px.scatter_matrix(
                        plot_data.select_dtypes(include=[np.number])
                    )
                else:
                    st.error("Unsupported chart type selected.")
                    fig = None

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")

        # Advanced Visualization Options
        st.subheader("Advanced Visualization")
        if st.checkbox("Show Correlation Heatmap"):
            numeric_df = plot_data.select_dtypes(include=['number'])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No numeric data available for correlation heatmap.")
    else:
        st.error("No columns available for plotting.")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Predict Trends'
        st.session_state.dummy_var += 1

def page5_predict_trends():
    st.title("Step 5: Predict Trends")
    st.write("Use linear regression to predict future trends.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if "Date" in df.columns and not df['Date'].isnull().all():
        selected_numeric_column = st.selectbox(
            "Select numeric column for prediction",
            numeric_columns
        )
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date', selected_numeric_column], inplace=True)
            df.sort_values('Date', inplace=True)

            if df.empty:
                st.error("No data available after cleaning. Please check your data.")
            else:
                df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

                model = LinearRegression()
                X = df[['Date_ordinal']]
                y = df[selected_numeric_column]
                model.fit(X, y)

                future_dates = pd.date_range(df['Date'].max(), periods=30)
                future_ordinal = future_dates.map(
                    pd.Timestamp.toordinal
                ).values.reshape(-1, 1)
                predictions = model.predict(future_ordinal)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=y,
                        mode='lines',
                        name='Actual'
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines',
                        name='Prediction',
                        line=dict(dash='dash')
                    )
                )
                fig.update_layout(
                    title="Trend Prediction",
                    xaxis_title="Date",
                    yaxis_title=selected_numeric_column
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during trend prediction: {e}")
    else:
        st.error("The dataset does not contain a valid 'Date' column.")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Clustering'
        st.session_state.dummy_var += 1

def page6_clustering():
    st.title("Step 6: Clustering")
    st.write("Segment your data into clusters.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    num_clusters = st.slider(
        "Select number of clusters",
        2, 10, 3,
        help="Choose the number of clusters for K-Means."
    )
    numeric_data = df.select_dtypes(include=['number']).dropna()
    if not numeric_data.empty:
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(numeric_data)
            df['Cluster'] = clusters
            st.write(df)
            st.session_state.df = df

            # Visualize Clusters
            if len(numeric_data.columns) >= 2:
                x_col, y_col = numeric_data.columns[:2]
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color='Cluster',
                    title="Cluster Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns to visualize clusters.")
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")
    else:
        st.error("No numeric data available for clustering.")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Custom Dashboard'
        st.session_state.dummy_var += 1

def page7_custom_dashboard():
    st.title("Step 7: Custom Dashboard")
    st.write("Create and save your own dashboard configurations.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    # Check if visualization parameters are in session_state
    required_params = ['data_source', 'chart_type', 'x_column', 'y_column']
    missing_params = [
        param for param in required_params if param not in st.session_state
    ]

    if missing_params:
        st.info("Please set up your visualization parameters in the Visualization step.")
    else:
        # Retrieve parameters from session_state
        data_source = st.session_state['data_source']
        chart_type = st.session_state['chart_type']
        x_column = st.session_state['x_column']
        y_column = st.session_state['y_column']

        # Save current configuration
        dashboard_name = st.text_input("Enter a name for your dashboard")
        if st.button("Save Dashboard"):
            config = {
                'data_source': data_source,
                'chart_type': chart_type,
                'x_column': x_column,
                'y_column': y_column
            }
            st.session_state[f'dashboard_{dashboard_name}'] = config
            st.success(f"Dashboard '{dashboard_name}' saved!")

        # Load saved configuration
        saved_dashboards = [
            key.replace('dashboard_', '')
            for key in st.session_state.keys()
            if key.startswith('dashboard_')
        ]
        if saved_dashboards:
            selected_dashboard = st.selectbox(
                "Load a saved dashboard",
                saved_dashboards
            )
            if st.button("Load Dashboard"):
                config = st.session_state[f'dashboard_{selected_dashboard}']
                data_source = config['data_source']
                chart_type = config['chart_type']
                x_column = config['x_column']
                y_column = config['y_column']
                st.success(f"Dashboard '{selected_dashboard}' loaded!")

                # Recreate the plot
                if data_source == "Original Data":
                    plot_data = df.copy()
                else:
                    plot_data = st.session_state.get('filtered_df', df)
                try:
                    if chart_type == "Line Chart":
                        fig = px.line(plot_data, x=x_column, y=y_column)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(plot_data, x=x_column, y=y_column)
                    # Add other chart types as necessary
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred while loading the dashboard: {e}")
        else:
            st.info("No dashboards saved yet.")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'NLP Query'
        st.session_state.dummy_var += 1

def page8_nlp_query():
    st.title("Step 8: NLP Query")
    st.write("Query your data using natural language.")
    df = st.session_state.df.copy()

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    user_query = st.text_input(
        "Enter your query",
        help="Example: 'Show me total sales where region is Europe and sales > 1000'"
    )

    if st.button("Run Query"):
        try:
            # Basic NLP processing to convert natural language to pandas query
            def nlp_to_pandas_query(query, df_columns):
                query = query.lower()
                # Extract conditions after 'where'
                where_match = re.search(r'where (.*)', query)
                if where_match:
                    conditions = where_match.group(1)
                    # Replace natural language operators with pandas operators
                    conditions = conditions.replace(" and ", " & ")
                    conditions = conditions.replace(" or ", " | ")
                    conditions = re.sub(r'is\s+([^\s]+)', r'== "\1"', conditions)
                    conditions = re.sub(r'equals\s+([^\s]+)', r'== "\1"', conditions)
                    conditions = re.sub(r'not equal to\s+([^\s]+)', r'!= "\1"', conditions)
                    conditions = re.sub(r'greater than\s+([^\s]+)', r'> \1', conditions)
                    conditions = re.sub(r'less than\s+([^\s]+)', r'< \1', conditions)
                    # Handle column names
                    for col in df_columns:
                        conditions = re.sub(r'\b{}\b'.format(col.lower()), f'`{col}`', conditions)
                    return conditions
                else:
                    return ''

            pandas_query = nlp_to_pandas_query(user_query, df.columns)
            if pandas_query:
                st.write(f"Generated pandas query: `{pandas_query}`")
                result = df.query(pandas_query)
            else:
                st.write("No conditions found in the query. Displaying all data.")
                result = df.copy()
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred while processing the query: {e}")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Real-Time Data'
        st.session_state.dummy_var += 1


def page9_real_time_data():
    st.title("Step 9: Real-Time Data Updates")
    st.write("Analyze streaming data in real-time.")
    df = st.session_state.df.copy()

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    # Check if 'Date' column exists and is valid
    if "Date" not in df.columns or df['Date'].isnull().all():
        st.error("The dataset does not contain a valid 'Date' column.")
        return

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_columns:
        st.error("No numeric columns available for real-time analysis.")
        return

    selected_numeric_column = st.selectbox(
        "Select numeric column for real-time analysis",
        numeric_columns
    )

    # Simulate real-time data updates
    if st.button("Start Real-Time Analysis"):
        placeholder = st.empty()
        df_rt = df.copy()

        for i in range(20):  # Update 20 times
            # Simulate new data
            new_row = df_rt.iloc[-1:].copy()
            new_row['Date'] = new_row['Date'] + pd.Timedelta(minutes=1)
            new_row[selected_numeric_column] = new_row[selected_numeric_column] + np.random.randn()
            df_rt = pd.concat([df_rt, new_row], ignore_index=True)

            # Update the chart
            fig = px.line(
                df_rt,
                x='Date',
                y=selected_numeric_column,
                title="Real-Time Data"
            )
            placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(1)  # Wait for 1 second


def page10_export_data():
    st.title("Step 10: Export Data")
    st.write("Download your data and reports.")
    df = st.session_state.df

    if df.empty:
        st.warning("Please upload and prepare data before proceeding.")
        return

    # Export Filtered Data
    if st.button("Export Filtered Data to CSV"):
        csv = st.session_state.get('filtered_df', df).to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Export Data to Excel
    if st.button("Export Data to Excel"):
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(excel_buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.ms-excel;base64,{b64}" download="data.xlsx">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)

    # Export Charts as Images
    st.write(
        "To save charts, right-click on the chart and select 'Save image as...'. "
        "Alternatively, you can export charts to PDF below."
    )

    # Generate PDF Report
    if st.button("Generate PDF Report"):
        try:
            pdf_output = BytesIO()
            c = canvas.Canvas(pdf_output)
            c.setFont("Helvetica", 12)
            c.drawString(200, 800, "Summary Report")
            c.setFont("Helvetica", 10)
            text = c.beginText(50, 780)
            for line in df.describe().to_string().split('\n'):
                text.textLine(line)
            c.drawText(text)
            c.save()
            b64 = base64.b64encode(pdf_output.getvalue()).decode()
            href = (
                f'<a href="data:application/pdf;base64,{b64}" '
                'download="summary_report.pdf">Download PDF Report</a>'
            )
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while generating the PDF report: {e}")

    if st.button("Next", key=f"next_{st.session_state.current_page}"):
        st.session_state.current_page = 'Help'
        st.session_state.dummy_var += 1

def page11_help():
    st.title("Step 11: Help and Documentation")
    st.write("Guidance on how to use the dashboard.")

    # Interactive Tutorials and Documentation
    st.markdown("""
    **How to Use This Dashboard**

    - **Step 1: Data Upload**: Start by uploading your dataset in CSV, Excel, or Parquet format.
    - **Step 2: Data Cleaning**: Identify and handle missing values, and convert data types if necessary.
    - **Step 3: Data Analysis**: Filter and sort your data to focus on what's important.
    - **Step 4: Visualization**: Create interactive charts to visualize trends and patterns.
    - **Step 5: Predict Trends**: Use linear regression to forecast future values.
    - **Step 6: Clustering**: Segment your data into clusters for deeper insights.
    - **Step 7: Custom Dashboard**: Save and load your dashboard configurations.
    - **Step 8: NLP Query**: Query your data using natural language sentences.
    - **Step 9: Real-Time Data**: Simulate and analyze streaming data in real-time.
    - **Step 10: Export Data**: Download your filtered data, charts, and reports.
    - **Step 11: Help**: Access guidance and FAQs.

    **Frequently Asked Questions**

    - **Q**: What file formats are supported?
      **A**: CSV, Excel, and Parquet files are supported.

    - **Q**: How do I handle missing values?
      **A**: Go to the 'Data Cleaning' section to fill or remove missing values.

    - **Q**: Can I save my analysis?
      **A**: Yes, use the 'Custom Dashboard' section to save and load configurations.

    **Need more help?** Contact support at [support@example.com](mailto:support@example.com).
    """)

    st.success("You have completed all the steps!")

# Main execution
missing_steps = []
if st.session_state.current_page == 'Data Cleaning' and st.session_state.df.empty:
    missing_steps.append("Data Upload")
elif st.session_state.current_page == 'Data Analysis' and st.session_state.df.empty:
    missing_steps.append("Data Upload and Data Cleaning")
elif st.session_state.current_page == 'Visualization' and st.session_state.df.empty:
    missing_steps.append("Data Upload")
elif st.session_state.current_page == 'Custom Dashboard' and st.session_state.df.empty:
    missing_steps.append("Data Upload and Visualization")

if missing_steps:
    st.warning(
        f"Please complete the following steps before accessing "
        f"'{st.session_state.current_page}': {', '.join(missing_steps)}."
    )
else:
    # Call the appropriate page function
    page_functions[st.session_state.current_page]()
