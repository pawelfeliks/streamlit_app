import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple Data Dashboard")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        st.subheader("Data Summary")
        st.write(df.describe())

        st.subheader("Filter Data")
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select column to filter by", columns)
        unique_values = df[selected_column].unique()
        selected_value = st.selectbox("Select value", unique_values)

        filtered_df = df[df[selected_column] == selected_value]
        st.write(filtered_df)

        st.subheader("Pivot Table")
        pivot_index = st.selectbox("Select index for pivot table", columns)
        pivot_values = st.multiselect("Select values for pivot table", columns)
        pivot_aggfunc = st.selectbox("Select aggregation function", ['mean', 'sum', 'count'])

        if pivot_values:
            try:
                pivot_table = pd.pivot_table(filtered_df, index=pivot_index, values=pivot_values, aggfunc=pivot_aggfunc)
                st.write(pivot_table)
            except Exception as e:
                st.error(f"An error occurred creating pivot table: {e}")

        st.subheader("Plot Data")
        chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Scatter Plot"])
        x_column = st.selectbox("Select x-axis column", columns)
        y_column = st.selectbox("Select y-axis column", columns)

        if st.button("Generate Plot"):
            if chart_type == "Line Chart":
                st.line_chart(filtered_df.set_index(x_column)[y_column])
            elif chart_type == "Bar Chart":
                fig, ax = plt.subplots()
                ax.bar(filtered_df[x_column], filtered_df[y_column])
                plt.xticks(rotation=45)
                st.pyplot(fig)
            elif chart_type == "Scatter Plot":
                fig, ax = plt.subplots()
                ax.scatter(filtered_df[x_column], filtered_df[y_column])
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Waiting on file upload...")
