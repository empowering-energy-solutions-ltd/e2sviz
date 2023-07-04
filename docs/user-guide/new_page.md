# Standard Data Process

::: src.data.standard_data_process.DataPrep
    handler: python
    options:
        members:
          - __post_init__
          - described_data
          - clean_data
          - statistics_of_data
          - concat
        show_root_heading: True
        show_source: True

::: src.data.standard_data_process.ColumnVizData
    handler: python
    options:
        members:
          - units
          - freq
          - column_name
          - get_x_label
          - get_y_label
          - get_title
          - get_ylim
          - plot_all
          - get_plotting_settings
        show_root_heading: True
        show_source: True
