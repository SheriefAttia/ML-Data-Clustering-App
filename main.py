import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import seaborn as sns

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(lambda x: (x - datetime(1970, 1, 1)).days).values.reshape(-1, 1)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Clustering Application")
        self.geometry("1000x1000")
        self.file_path = ""
        self.num_clusters = 2
        self.cluster_method = "KMeans"
        self.group_by_vars = {}
        self.sort_column_var = tk.StringVar(self)
        self.sort_order_var = tk.StringVar(self)
        self.filter_column_var = tk.StringVar(self)
        self.create_widgets()

    def create_widgets(self):
        self.group_by_vars = {}  # A dictionary to store checkbox variables
        row = 14  # Starting row for checkboxes

        ttk.Label(self, text="Select Data File:").grid(column=0, row=0, padx=10, pady=5, sticky="w")
        self.file_entry = ttk.Entry(self, width=50)
        self.file_entry.grid(column=1, row=0, padx=10, pady=5, sticky="we")
        ttk.Button(self, text="Browse...", command=self.browse_file).grid(column=2, row=0, padx=10, pady=5)

        ttk.Button(self, text="Load Data", command=self.load_and_update_gui).grid(column=0, row=1, columnspan=3,
                                                                                  pady=10)

        ttk.Label(self, text="Number of Clusters:").grid(column=0, row=1, padx=10, pady=5, sticky="w")
        self.num_clusters_spinbox = ttk.Spinbox(self, from_=2, to=10, width=5)
        self.num_clusters_spinbox.grid(column=1, row=1, padx=10, pady=5, sticky="w")

        ttk.Label(self, text="Missing Values Strategy:").grid(column=0, row=2, padx=10, pady=5, sticky="w")
        self.missing_values_combobox = ttk.Combobox(self, values=["mean", "median", "most_frequent", "constant"])
        self.missing_values_combobox.grid(column=1, row=2, padx=10, pady=5, sticky="w")
        self.missing_values_combobox.current(0)

        ttk.Label(self, text="DBSCAN eps:").grid(column=0, row=3, padx=10, pady=5, sticky="w")
        self.dbscan_eps_entry = ttk.Entry(self)
        self.dbscan_eps_entry.grid(column=1, row=3, padx=10, pady=5)
        self.dbscan_eps_entry.insert(0, "0.5")

        ttk.Label(self, text="DBSCAN min_samples:").grid(column=0, row=4, padx=10, pady=5, sticky="w")
        self.dbscan_min_samples_spinbox = ttk.Spinbox(self, from_=1, to=100, width=5)
        self.dbscan_min_samples_spinbox.grid(column=1, row=4, padx=10, pady=5)
        self.dbscan_min_samples_spinbox.insert(0, "5")

        ttk.Label(self, text="Clustering Method:").grid(column=0, row=5, padx=10, pady=5, sticky="w")
        self.method_combobox = ttk.Combobox(self, values=["KMeans", "DBSCAN", "Agglomerative"], state="readonly")
        self.method_combobox.grid(column=1, row=5, padx=10, pady=5, sticky="w")
        self.method_combobox.current(0)

        ttk.Label(self, text="Variance Threshold:").grid(column=0, row=6, padx=10, pady=5, sticky="w")
        self.variance_threshold_entry = ttk.Entry(self)
        self.variance_threshold_entry.grid(column=1, row=6, padx=10, pady=5)
        self.variance_threshold_entry.insert(0, "0.0")

        ttk.Label(self, text="SVD Components:").grid(column=0, row=7, padx=10, pady=5, sticky="w")
        self.svd_components_spinbox = ttk.Spinbox(self, from_=1, to=100, width=5)
        self.svd_components_spinbox.grid(column=1, row=7, padx=10, pady=5)
        self.svd_components_spinbox.insert(0, "2")

        ttk.Label(self, text="Sort By:").grid(column=0, row=8, padx=10, pady=5, sticky="w")
        self.sort_column_var = tk.StringVar(self)
        self.sort_column_var.set("")
        sort_column_dropdown = ttk.OptionMenu(self, self.sort_column_var, "")
        sort_column_dropdown.grid(column=1, row=8, padx=10, pady=5)

        ttk.Label(self, text="Sort Order:").grid(column=0, row=9, padx=10, pady=5, sticky="w")
        self.sort_order_var = tk.StringVar(self)
        self.sort_order_var.set("Ascending")
        ttk.Radiobutton(self, text="Ascending", variable=self.sort_order_var, value="Ascending").grid(column=1, row=9,
                                                                                                      padx=10, pady=5,
                                                                                                      sticky="w")
        ttk.Radiobutton(self, text="Descending", variable=self.sort_order_var, value="Descending").grid(column=2,
                                                                                                        row=9, padx=10,
                                                                                                        pady=5,
                                                                                                        sticky="w")

        ttk.Label(self, text="Filter By:").grid(column=0, row=10, padx=10, pady=5, sticky="w")
        self.filter_column_var = tk.StringVar(self)
        filter_column_dropdown = ttk.Combobox(self, textvariable=self.filter_column_var)
        filter_column_dropdown.grid(column=1, row=10, padx=10, pady=5, sticky="we")

        self.filter_value_entry = ttk.Entry(self)
        self.filter_value_entry.grid(column=2, row=10, padx=10, pady=5, sticky="we")

        ttk.Button(self, text="Apply Filtering", command=self.apply_filtering).grid(column=0, row=11, columnspan=3,
                                                                                    pady=10)

        ttk.Label(self, text="Group By:").grid(column=0, row=12, padx=10, pady=5, sticky="w")
        # Add dropdowns or checkboxes for column selection

        ttk.Button(self, text="Apply Grouping", command=self.apply_grouping).grid(column=0, row=13, columnspan=3,
                                                                                  pady=10)

        # Add dropdowns or checkboxes for pivot table settings
        ttk.Label(self, text="Pivot Table:").grid(column=0, row=14, padx=10, pady=5, sticky="w")

        self.pivot_index_var = tk.StringVar(self)
        index_dropdown = ttk.Combobox(self, textvariable=self.pivot_index_var)
        index_dropdown.grid(column=1, row=14, padx=10, pady=5)

        self.pivot_columns_var = tk.StringVar(self)
        columns_dropdown = ttk.Combobox(self, textvariable=self.pivot_columns_var)
        columns_dropdown.grid(column=2, row=14, padx=10, pady=5)

        self.pivot_values_var = tk.StringVar(self)
        values_dropdown = ttk.Combobox(self, textvariable=self.pivot_values_var)
        values_dropdown.grid(column=1, row=15, padx=10, pady=5)

        self.pivot_agg_var = tk.StringVar(self)
        agg_dropdown = ttk.Combobox(self, textvariable=self.pivot_agg_var,
                                    values=["mean", "median", "sum", "count"])
        agg_dropdown.grid(column=2, row=15, padx=10, pady=5)
        agg_dropdown.current(0)  # Default to mean

        ttk.Button(self, text="Generate Pivot Table", command=self.generate_pivot_table).grid(
            column=0, row=16, columnspan=3, pady=10)

        ttk.Button(self, text="Perform EDA", command=self.perform_eda).grid(column=0, row=16, columnspan=3, pady=10)

        ttk.Button(self, text="Start Machine Learning", command=self.start_machine_learning).grid(column=0, row=17,
                                                                                                  columnspan=3, pady=10)

        ttk.Button(self, text="Apply Sorting", command=self.apply_sorting).grid(column=0, row=18, columnspan=3, pady=10)

        ttk.Button(self, text="Start Clustering", command=self.start_clustering).grid(column=0, row=19, columnspan=3, pady=10)

    def load_and_update_gui(self):
        self.file_path = self.file_entry.get()  # Get file path from entry widget
        if self.file_path:
            self.load_data(self.file_path)
            self.update_gui_elements()

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, self.file_path)

    def load_data(self, file_path):
        try:
            self.df = pd.read_excel(file_path)
            messagebox.showinfo("Info", "Data loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def update_gui_elements(self):
        if self.df is not None:
            columns = self.df.columns.tolist()
            # Update all checkboxes with column names
            row = 14  # Resetting the row variable
            for column in columns:
                var = tk.BooleanVar(self)
                ttk.Checkbutton(self, text=column, variable=var).grid(column=1, row=row, padx=10, pady=5, sticky="w")
                self.group_by_vars[column] = var
                row += 1

            # Update pivot table settings comboboxes
            self.pivot_index_var.set(','.join(columns))  # Set index columns to all columns by default
            self.pivot_columns_var.set(','.join(columns))  # Set columns columns to all columns by default
            self.pivot_values_var.set(columns[0])  # Set values column to the first column by default

    def get_pivot_setting(self, setting_type):
        if setting_type == 'index':
            return self.pivot_index_var.get().split(',')  # Allow multiple index columns
        elif setting_type == 'values':
            return self.pivot_values_var.get()
        elif setting_type == 'columns':
            return self.pivot_columns_var.get()
        elif setting_type == 'aggfunc':
            agg_str = self.pivot_agg_var.get()
            return np.mean if agg_str == 'mean' else np.median if agg_str == 'median' else np.sum if agg_str == 'sum' else len

    def apply_filtering(self):
        filter_criteria = self.get_filter_criteria_from_gui()
        if filter_criteria:
            try:
                # Apply filters based on specific criteria
                self.df = self.df[filter_criteria]
                # Update your GUI to reflect filtered data (if needed)
            except (ValueError, KeyError) as e:
                messagebox.showerror("Filtering Error", f"Error applying filter: {e}")

    def apply_grouping(self):
        group_by_columns = []
        for col, var in self.group_by_vars.items():
            if var.get():
                group_by_columns.append(col)

        if group_by_columns:
            try:
                grouped_df = self.df.groupby(group_by_columns).size().reset_index(name='count')
                # Update the GUI to display the grouped_df (e.g., Treeview)
            except (ValueError, KeyError) as e:
                messagebox.showerror("Grouping Error", f"Error applying grouping: {e}")

    def display_descriptive_stats(self, descriptive_stats):
        stats_window = tk.Toplevel(self)
        stats_window.title("Descriptive Statistics")

        tree = ttk.Treeview(stats_window)
        tree['columns'] = descriptive_stats.columns.tolist()
        tree.column("#0", width=0, stretch=tk.NO)  # Hide the initial default column

        # Column headings
        for col in tree['columns']:
            tree.heading(col, text=col)

        # Insert data as rows
        for index, row in descriptive_stats.iterrows():
            tree.insert("", tk.END, values=row.tolist())

        tree.pack()

    def generate_pivot_table(self):
        try:
            if self.df is not None:
                index_columns = []  # List to store selected index columns
                columns_column = []  # List to store selected columns columns

                # Retrieve selected index columns
                for col, var in self.group_by_vars.items():
                    if var.get():
                        index_columns.append(col)

                # Retrieve selected columns columns
                for col, var in self.group_by_vars.items():
                    if var.get():
                        columns_column.append(col)

                values_column = self.pivot_values_var.get()  # Retrieve selected values column
                agg_function = self.get_pivot_setting('aggfunc')  # Retrieve selected aggregation function
                pivot_table = self.df.pivot_table(index=index_columns,
                                                  columns=columns_column,
                                                  values=values_column,
                                                  aggfunc=agg_function)

                # Display the pivot table
                print(pivot_table)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate pivot table: {e}")

    def perform_eda(self):
        if self.df is not None:
            descriptive_stats = self.df.describe(include='all')
            self.display_descriptive_stats(descriptive_stats)
        else:
            messagebox.showwarning("Warning", "No data loaded for EDA.")

    def start_machine_learning(self):
        messagebox.showinfo("Machine Learning", "Machine learning functionality not implemented yet.")

    def apply_sorting(self):
        sort_column = self.sort_column_var.get()
        sort_order = self.sort_order_var.get()
        if sort_column:
            try:
                self.df.sort_values(by=sort_column, ascending=(sort_order == 'Ascending'), inplace=True)
                # Update your GUI to reflect sorted data (if needed)
            except KeyError:
                messagebox.showerror("Sorting Error", "Invalid column selected for sorting.")
                
    def start_machine_learning(self):
        # Get machine learning settings from GUI
        # Split data into training and testing sets using sklearn.model_selection.train_test_split
        # Choose appropriate algorithm based on user selection
        # Fit the selected model, make predictions, and evaluate the model using sklearn.metrics
        pass

    def start_clustering(self):
        num_clusters_value = self.num_clusters_spinbox.get()
        if num_clusters_value:  # Check if the value is not empty
            self.num_clusters = int(num_clusters_value)
            self.cluster_method = self.method_combobox.get()
            if self.file_path:
                try:
                    self.cluster_data()
                except Exception as e:
                    messagebox.showerror("Error", str(e))
            else:
                messagebox.showinfo("Info", "Please select a file to proceed.")
        else:
            messagebox.showwarning("Warning", "Please enter a valid number of clusters.")

    def cluster_data(self):
        self.df = pd.read_excel(self.file_path)
        self.df.dropna(axis=1, how='all', inplace=True)

        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        date_features = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        # Update numeric_transformer based on the missing values strategy selected by the user
        missing_values_strategy = self.missing_values_combobox.get()
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy=missing_values_strategy)),
                                              ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        date_transformer = Pipeline(steps=[('date_extractor', DateTransformer())])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', date_transformer, date_features)
        ])

        # Reading dynamic configurations for feature selection and dimensionality reduction
        variance_threshold = float(self.variance_threshold_entry.get())
        n_components = int(self.svd_components_spinbox.get())
        feature_selection = VarianceThreshold(threshold=variance_threshold)
        svd = TruncatedSVD(n_components=n_components)

        # Configuring clustering algorithm based on user selection
        if self.cluster_method == "KMeans":
            clusterer = KMeans(n_clusters=self.num_clusters, random_state=42)
        elif self.cluster_method == "DBSCAN":
            eps = float(self.dbscan_eps_entry.get())
            min_samples = int(self.dbscan_min_samples_spinbox.get())
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif self.cluster_method == "Agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=self.num_clusters)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selection),
            ('svd', svd),
            ('cluster', clusterer)
        ])

        self.df['Cluster'] = pipeline.fit_predict(self.df)
        # Compute SVD components for all cases right after fitting the pipeline
        preprocessed_data = pipeline.named_steps['preprocessor'].transform(self.df)
        selected_features = pipeline.named_steps['feature_selection'].transform(preprocessed_data)
        svd_components = pipeline.named_steps['svd'].transform(selected_features)

        # Show silhouette score for KMeans and Agglomerative Clustering
        if self.cluster_method in ["KMeans", "Agglomerative"]:
            score = silhouette_score(svd_components, self.df['Cluster'])
            messagebox.showinfo("Cluster Quality", f"Silhouette Score: {score:.2f}")

        self.visualize_clusters(svd_components, self.df['Cluster'])
        self.save_clustered_data(self.df)

    def visualize_clusters(self, svd_components, clusters):
        if svd_components.shape[1] >= 3:
            # 3D Visualization if we have enough components
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(svd_components[:, 0], svd_components[:, 1], svd_components[:, 2], c=clusters, cmap='viridis')
            plt.title("3D Cluster Visualization")
        else:
            # 2D Visualization
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=svd_components[:, 0], y=svd_components[:, 1], hue=clusters, palette="viridis")
            plt.title("Cluster Visualization")
            plt.xlabel("SVD Component 1")
            plt.ylabel("SVD Component 2")
        plt.show()

    def save_clustered_data(self, df):
        output_file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                        title="Save the clustered data as")
        if output_file_path:
            try:
                df.to_excel(output_file_path, index=False)
                messagebox.showinfo("Success", "The clustered data has been saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save the file: {e}")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
