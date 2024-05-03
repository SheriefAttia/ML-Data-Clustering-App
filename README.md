
# ML Data Clustering App

Welcome to the ML Data Clustering App, a robust tool designed to facilitate data analysis through clustering techniques. This application leverages popular algorithms like KMeans, DBSCAN, and Agglomerative Clustering, wrapped in a user-friendly GUI built with Tkinter. Whether you are a data scientist, a student, or just someone curious about data clustering, this tool is designed to help you understand and apply clustering algorithms to real-world datasets effectively.

## Key Features

- **Excel Integration**: Directly load your data from Excel files, streamlining the workflow for non-programmers or those preferring Excel for data storage.
- **Comprehensive Data Preprocessing**: Includes options for handling missing data, normalizing and scaling data, and encoding categorical variables to ensure the data is primed for effective clustering.
- **Multiple Clustering Algorithms**: Choose from KMeans, DBSCAN, or Agglomerative Clustering based on your specific dataset characteristics and clustering needs.
- **Interactive GUI**: Control your data analysis process with an easy-to-use interface that allows dynamic parameter adjustments.
- **Advanced Visualization**: Generate 2D and 3D plots to visualize clustering results, helping you better understand the distribution and grouping of your data.
- **Export Capabilities**: Easily save your results and figures to Excel and PNG formats, allowing for further analysis or presentation.

## Installation Guide

This section covers the prerequisites and steps to get the application up and running on your machine.

### Prerequisites

Ensure you have the following installed:
- Python 3.x: [Download Python](https://www.python.org/downloads/)
- Pip: Python's package installer (included with Python 3.4 and later).

### Setup Environment

It's recommended to use a virtual environment to avoid conflicts with other package versions you may have installed:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/SheriefAttia/ML-Data-Clustering-App.git
cd ML-Data-Clustering-App
```

### Install Dependencies

Install all required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Use

Follow these steps to run the application:

1. **Start the Application**:
    ```bash
    python yourscript.py
    ```

2. **Load Data**:
    - Use the 'Browse...' button in the GUI to select your Excel file.

3. **Set Clustering Parameters**:
    - Adjust the parameters such as the number of clusters or the eps and min_samples for DBSCAN through the GUI controls.

4. **Perform Clustering**:
    - Click the 'Start Clustering' button to process your data according to the selected algorithm and parameters.

5. **View and Save Results**:
    - Visualize the clustering output directly in the GUI and use the 'Save' button to export the results to Excel or as images.

## Contributing

Contributions to this project are welcome! Here's how you can contribute:
- Fork the repository on GitHub.
- Clone your forked repository to your local machine.
- Create a new feature branch (`git checkout -b my-new-feature`).
- Make your changes and commit them (`git commit -am 'Add some feature'`).
- Push the branch to GitHub (`git push origin my-new-feature`).
- Submit a pull request on GitHub from your feature branch to the main project's `main` branch.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- A heartfelt thanks to the contributors of the open-source libraries used in this project.
- Special thanks to the Python community for continuous support and inspiration.
