e2sviz
==============================

E2SViz is a data visualization package that allows data preprocessing, `e2sviz.data.standard_data_process.DataPrep` & `e2sviz.data.standard_data_process.DataManip`, and data visualisation, `e2sviz.visualization.visualize.DataViz`.

Examples of how to use e2svizs' functionality can be found in `e2sviz_example.ipynb` which uses demo data `example_consumption_data.csv`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    │   │
    │   ├── e2sviz_example.ipynb  <- Example notebook containing all the functions and how to use them.
    │   └── example_consumption_data.csv  <- Example dataset for use with the example notebook.
    │
    ├── e2sviz             <- Package import when using e2sviz as an imported pacakge.
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts for data preperation and manipulations.
    │   │   ├── data_preperation.py    <- Contains cleaner functions used in the DataPrep
    │   │   │   
    │   │   ├── functions.py   <- General functions used by manipularor and cleaner functions.
    │   │   │   
    │   │   ├── manipulations.py   <- Classes and functions used by DataManip.
    │   │   │   
    │   │   └── standard_data_process.py   <- DataPrep, Metadata & DataManip classes.
    │   │
    │   ├── structure       <- Schemas and enums files for e2sviz
    │   │   │   
    │   │   ├── datetime_schema.py   <- Contains schemas for datetime variables
    │   │   │   
    │   │   ├── enums.py   <- Contains project enum values
    │   │   │   
    │   │   └── viz_schema.py   <- Contains plot schemas used by Visualization
    │   │
    │   └── visualization    <- Contains the plot styles and visualization class and functions.
    │       │   
    │       ├── plot_styles.py   <- Contains Matplotlib & Plotly plot classes that holds the figure axes passed to the DataViz obj
    │       │   
    │       └── visualize.py   <- DataViz object file
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
