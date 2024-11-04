import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from io import BytesIO
from reportlab.pdfgen import canvas

st.title("Advanced Data Dashboard")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        # Data Summary
        st.subheader("Data Summary")
        st.write(df.describe())

        # Data Cleaning Section
        st.subheader("Data Cleaning")
        if st.checkbox("Show missing values summary"):
            st.write(df.isna().sum())
        if st.button("Fill missing values with mean"):
            df.fillna(df.mean(), inplace=True)
            st.success("Missing values filled with mean!")

        # Interactive Filtering Section
        st.subheader("Filter Data")
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select column to filter by", columns)
        unique_values = df[selected_column].unique()
        selected_value = st.selectbox("Select value", unique_values)
        filtered_df = df[df[selected_column] == selected_value]

        # Numeric Filtering with Slider
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            selected_numeric_column = st.selectbox("Select a numeric column to filter", numeric_columns)
            min_val, max_val = st.slider(f"Select range for {selected_numeric_column}",
                                         float(df[selected_numeric_column].min()),
                                         float(df[selected_numeric_column].max()),
                                         (float(df[selected_numeric_column].min()), float(df[selected_numeric_column].max())))
            filtered_df = filtered_df[(filtered_df[selected_numeric_column] >= min_val) & (filtered_df[selected_numeric_column] <= max_val)]
            st.write(filtered_df)

        # Sort Data Section
        st.subheader("Sort Data")
        sort_column = st.selectbox("Select column to sort by", columns)
        sort_order = st.radio("Select sort order", ("Ascending", "Descending"))
        sorted_df = filtered_df.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
        st.write(sorted_df)

        # Add/Edit/Delete Rows Section
        st.subheader("Add/Edit/Delete Rows")
        new_row_data = {}
        st.write("### Add a new row")
        for col in columns:
            new_row_data[col] = st.text_input(f"Enter value for {col}")
        if st.button("Add Row"):
            new_row = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_row], ignore_index=True)
            st.success("New row added!")
            st.write(df.tail())

        # Editable Table
        st.subheader("Edit Data Table")
        edited_df = st.data_editor(df, num_rows="dynamic")
        if st.button("Save Changes"):
            df = edited_df
            st.success("Changes saved!")

        # Pivot Table Section
        st.subheader("Pivot Table")
        pivot_index = st.multiselect("Select index for pivot table", columns)
        pivot_columns = st.multiselect("Select columns for pivot table", columns)
        pivot_values = st.multiselect("Select values for pivot table", columns)
        pivot_aggfunc = st.selectbox("Select aggregation function", ['mean', 'sum', 'count', 'min', 'max', 'median', 'std'])

        if pivot_values:
            try:
                pivot_table = pd.pivot_table(filtered_df, index=pivot_index, columns=pivot_columns, values=pivot_values, aggfunc=pivot_aggfunc)
                st.write(pivot_table)
                
                # Option to plot pivot table
                if st.checkbox("Plot Pivot Table"):
                    plot_pivot_chart_type = st.selectbox("Select chart type for pivot table", ["Line Chart", "Bar Chart", "Heatmap"])
                    if plot_pivot_chart_type == "Line Chart":
                        st.line_chart(pivot_table)
                    elif plot_pivot_chart_type == "Bar Chart":
                        st.bar_chart(pivot_table)
                    elif plot_pivot_chart_type == "Heatmap":
                        fig, ax = plt.subplots()
                        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', ax=ax)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred creating pivot table: {e}")

        # Correlation Heatmap
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # New Plot Data Section
        st.subheader("Plot Data")

        # Select data source
        data_source = st.selectbox("Select data source for plotting", ["Original Data", "Filtered Data", "Pivot Table"])

        # Select chart type
        all_chart_types = ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Area Chart", "Pie Chart", "Heatmap", "Pairplot", "Violin Plot", "Bubble Chart", "Radar Chart", "Boxen Plot", "Swarm Plot", "Strip Plot", "Count Plot"]
        chart_type = st.selectbox("Select chart type", all_chart_types)

        # Depending on data source, get the data
        if data_source == "Original Data":
            plot_data = df
        elif data_source == "Filtered Data":
            plot_data = filtered_df
        elif data_source == "Pivot Table":
            if 'pivot_table' in locals():
                plot_data = pivot_table.reset_index()
            else:
                st.error("No pivot table available for plotting.")
                plot_data = None
        else:
            plot_data = df

        if plot_data is not None:
            columns_plot = plot_data.columns.tolist()
            x_column = st.selectbox("Select x-axis column", columns_plot)
            y_column = st.selectbox("Select y-axis column", columns_plot)
            plot_button = st.button("Generate Plot")

            # Display the selected plot
            if plot_button:
                fig, ax = plt.subplots()
                if chart_type == "Line Chart":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]):
                        plot_data.set_index(x_column)[y_column].plot.line(ax=ax)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        st.error("Y-axis must be numeric for Line Chart.")
                elif chart_type == "Bar Chart":
                    plot_data.plot.bar(x=x_column, y=y_column, ax=ax)
                    plt.xticks(rotation=45)
                elif chart_type == "Scatter Plot":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]) and pd.api.types.is_numeric_dtype(plot_data[x_column]):
                        ax.scatter(plot_data[x_column], plot_data[y_column])
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        st.error("Both X and Y axes must be numeric for Scatter Plot.")
                elif chart_type == "Histogram":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]):
                        ax.hist(plot_data[y_column], bins=20)
                        plt.xlabel(y_column)
                        plt.ylabel("Frequency")
                    else:
                        st.error("Y-axis must be numeric for Histogram.")
                elif chart_type == "Box Plot":
                    sns.boxplot(data=plot_data, x=x_column, y=y_column, ax=ax)
                elif chart_type == "Area Chart":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]):
                        plot_data.set_index(x_column)[y_column].plot.area(ax=ax)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        st.error("Y-axis must be numeric for Area Chart.")
                elif chart_type == "Pie Chart":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]):
                        plot_data.groupby(x_column)[y_column].sum().plot.pie(ax=ax, autopct="%1.1f%%")
                        plt.ylabel("")
                    else:
                        st.error("Y-axis must be numeric for Pie Chart.")
                elif chart_type == "Heatmap":
                    if pd.api.types.is_numeric_dtype(plot_data[x_column]) and pd.api.types.is_numeric_dtype(plot_data[y_column]):
                        heatmap_data = pd.pivot_table(plot_data, values=y_column, index=x_column, aggfunc='mean')
                        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax)
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                    else:
                        st.error("Both X and Y axes must be numeric for Heatmap.")
                elif chart_type == "Pairplot":
                    sns.pairplot(plot_data, vars=[x_column, y_column])
                    st.pyplot()  # Pairplot needs separate rendering
                    st.stop()
                elif chart_type == "Violin Plot":
                    sns.violinplot(data=plot_data, x=x_column, y=y_column, ax=ax)
                elif chart_type == "Bubble Chart":
                    if pd.api.types.is_numeric_dtype(plot_data[y_column]) and pd.api.types.is_numeric_dtype(plot_data[x_column]):
                        size_column = st.selectbox("Select size column for Bubble Chart", columns_plot)
                        if pd.api.types.is_numeric_dtype(plot_data[size_column]):
                            ax.scatter(plot_data[x_column], plot_data[y_column], s=plot_data[size_column]*10, alpha=0.5)
                            plt.xlabel(x_column)
                            plt.ylabel(y_column)
                        else:
                            st.error("Size column must be numeric for Bubble Chart.")
                    else:
                        st.error("Both X and Y axes must be numeric for Bubble Chart.")
                elif chart_type == "Radar Chart":
                    categories = st.multiselect("Select categories for Radar Chart", columns_plot)
                    if categories:
                        data = plot_data[categories]
                        if all([pd.api.types.is_numeric_dtype(data[col]) for col in data.columns]):
                            labels = data.columns
                            num_vars = len(labels)
                            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                            angles += angles[:1]
                            values = data.mean().tolist()
                            values += values[:1]

                            fig = plt.figure(figsize=(6,6))
                            ax = fig.add_subplot(111, polar=True)
                            ax.plot(angles, values, 'o-', linewidth=2)
                            ax.fill(angles, values, alpha=0.25)
                            ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                            plt.title("Radar Chart")
                        else:
                            st.error("All selected categories must be numeric for Radar Chart.")
                    else:
                        st.error("Please select at least one category for Radar Chart.")
                elif chart_type == "Boxen Plot":
                    sns.boxenplot(data=plot_data, x=x_column, y=y_column, ax=ax)
                elif chart_type == "Swarm Plot":
                    sns.swarmplot(data=plot_data, x=x_column, y=y_column, ax=ax)
                elif chart_type == "Strip Plot":
                    sns.stripplot(data=plot_data, x=x_column, y=y_column, ax=ax)
                elif chart_type == "Count Plot":
                    sns.countplot(data=plot_data, x=x_column, ax=ax)
                else:
                    st.error("Invalid chart type selected.")
                st.pyplot(fig)

        # Trend Prediction Section
        st.subheader("Predict Trends")
        if "Date" in df.columns and selected_numeric_column in df.columns:
            try:
                date_column = pd.to_datetime(df['Date'], errors='coerce')
                df['Date_ordinal'] = date_column.apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
                df = df.dropna(subset=['Date_ordinal'])

                model = LinearRegression()
                X = np.array(df['Date_ordinal']).reshape(-1, 1)
                y = df[selected_numeric_column]
                model.fit(X, y)

                future_dates = pd.date_range(df['Date'].max(), periods=30)
                future_ordinal = future_dates.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)
                predictions = model.predict(future_ordinal)

                plt.plot(df['Date'], df[selected_numeric_column], label="Actual")
                plt.plot(future_dates, predictions, label="Prediction", linestyle="--")
                plt.legend()
                st.pyplot()
            except Exception as e:
                st.error(f"An error occurred during trend prediction: {e}")

        # Clustering Section
        st.subheader("Customer Segmentation using K-Means Clustering")
        num_clusters = st.slider("Select number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=num_clusters)
        numeric_data = df.select_dtypes(include=['number']).dropna()
        if not numeric_data.empty:
            kmeans.fit(numeric_data)
            df['Cluster'] = kmeans.labels_
            st.write(df)

        # Export Data Section
        st.subheader("Export Data")
        if st.button("Export Filtered Data to CSV"):
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

        if 'pivot_table' in locals():
            if st.button("Export Pivot Table to CSV"):
                pivot_csv = pivot_table.to_csv().encode('utf-8')
                st.download_button("Download Pivot Table CSV", pivot_csv, "pivot_table.csv", "text/csv")

        # Export Charts to PDF
        if st.button("Export Summary Report to PDF"):
            pdf_output = BytesIO()
            c = canvas.Canvas(pdf_output)
            c.setFont("Helvetica", 12)
            c.drawString(200, 800, "Summary Report")
            c.drawString(50, 780, "Data Summary")
            text = c.beginText(50, 760)
            text.setFont("Helvetica", 10)
            for line in df.describe().to_string().split('\n'):
                text.textLine(line)
            c.drawText(text)
            c.save()
            st.download_button("Download PDF Report", data=pdf_output.getvalue(), file_name="summary_report.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Waiting on file upload...")
