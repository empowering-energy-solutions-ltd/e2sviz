from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import pandas as pd

from e2sviz.structure import enums as viz_enums
from e2sviz.structure import viz_schema


class LibraryViz(Protocol):
  """
  Selects the visualisation library to be used (`e2sviz.visualization.plot_styles`)
  """
  container: Any

  def line_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    ...

  def stacked_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                   dict_kwargs: dict[str, dict[str, str]]):
    ...

  def corr_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    ...

  def bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    ...

  def dt_bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                  dict_kwargs: dict[str, dict[str, str]]):
    ...

  def box_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    ...

  def pie_chart(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    ...

  def show(self) -> Any:
    ...

  def save(self, save_path: Path) -> Any:
    ...


class MetaData(Protocol):
  """
  Stores the meta data and returns values for labeling plots (`e2sviz.data.standard_data_process`).
  """

  metadata: dict[str, dict[str, Any]]

  def units(self, col: str) -> viz_enums.UnitsSchema:
    ...

  def siunits(self, col: str) -> viz_enums.Prefix:
    ...

  @property
  def freq(self) -> viz_schema.FrequencySchema:
    ...

  def dtype(self, col: str) -> viz_enums.DataType:
    ...

  @property
  def get_x_label(self) -> str:
    ...

  def get_y_label(self, col: str) -> str:
    ...

  def get_title(self, col: str, category: str | None = None) -> str:
    ...

  def get_legend(self, col: str) -> str:
    ...


@dataclass
class DataViz:
  """
  Visualises the data.

  Attributes:
    data pd.DataFrame:
      The data to be visualised.
    metadata MetaData:
      The metadata of the data.
    viz_selector LibraryViz:
      The visualisation library to be used.
    plot_columns list[str]:
      The columns to be plotted.
    x str:
      The x-axis column.
    freq str:
      The frequency of the data.

  Methods:
    plot:
      Plots the data through the chosen viz_selector.
    show_viz:
      Shows the visualisation.
    save_figure:
      Saves the figure.
    structured_data:
      Returns the data in a structured format.
    create_dict_kwargs:
      Creates the list of kwargs for each column.
    _process_grouped_data:
      Process grouped data and pivot if needed.
    format_index:
      Format the index of the data based on the number of index columns in the metadata.
    _adjust_index:
      Adjust index to be a continuous variable.
    remove_index_cols:
      Remove the index columns from the data after they've been reset.
    pivot_data:
      Pivot the data based on the metadata grouped columns.
  
  """
  data: pd.DataFrame
  metadata: MetaData
  viz_selector: LibraryViz
  plot_columns: Optional[list[str]] = None
  x: Optional[str] = None
  freq: Optional[str] = None

  def __post_init__(self):
    if self.plot_columns is None:
      self.plot_columns: list[str] = self.data.columns.to_list()

  @property
  def plot_factory(self) -> dict[str, Callable]:
    """
    Returns the plot factory used for plotting the specific plot type.

    Returns
    -------
    `dict[str, Callable]`
      The plot factory used for plotting the specific plot type.
    """
    return {
        'line_plot': self.viz_selector.line_plot,
        'stacked_plot': self.viz_selector.stacked_plot,
        'corr_plot': self.viz_selector.corr_plot,
        'bar_plot': self.viz_selector.bar_plot,
        'dt_bar_plot': self.viz_selector.dt_bar_plot,
        'box_plot': self.viz_selector.box_plot,
        'pie_chart': self.viz_selector.pie_chart
    }

  def plot(self, plot_kind: str) -> None:
    """
    Plots the data through the chosen viz_selector.
    Options: `line_plot`, `stacked_plot`, `corr_plot`, `bar_plot`, `dt_bar_plot`, `box_plot`, `pie_chart`

    Parameters
    ----------
    `plot_kind` : `str`
      The kind of plot to be plotted.

    Returns
    -------
    `None`
      Figure is saved to the container object of the viz_selector.
    """
    plot_data = self.structured_data()
    dict_kwargs = self.create_dict_kwargs()
    if self.x is None:
      self.plot_factory[plot_kind](dataf=plot_data,
                                   plot_columns=self.plot_columns,
                                   dict_kwargs=dict_kwargs)
    else:
      self.plot_factory[plot_kind](dataf=plot_data,
                                   plot_columns=self.plot_columns,
                                   dict_kwargs=dict_kwargs,
                                   x=self.x)

  def show_viz(self) -> Any:
    """
    Shows the visualisation.

    Returns
    -------
    `Any`
      The visualisation currently matplotlib or plotly.
    """
    return self.viz_selector.show()

  def save_figure(self, save_path: Path) -> None:
    """
    Saves the figure.

    Parameters
    ----------
    `save_path` : `Path`
      The path to save the figure along.

    Returns
    -------
    `None`
      The figure is saved to the path.
    """
    self.viz_selector.save(save_path)

  def structured_data(self) -> pd.DataFrame:
    """
    Returns the data in a structured format.
    
    Returns
    -------
    `pd.DataFrame`
      The data in a structured format.
    """
    data_copy = self.data.copy()
    if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS]) > 0:
      data_copy = self._process_grouped_data(data_copy)
    return data_copy

  def create_dict_kwargs(self) -> dict[str, dict[str, Any]]:
    """
			Creates the dict of dicts of kwargs for each column.

			Returns
			-------
			`dict[str, dict[str, Any]]`
				The dict of dicts of kwargs for each column.
			"""
    dict_kwargs = {}
    x_label = self.metadata.get_x_label
    if self.freq:
      x_label = f"Datetime ({self.freq})"
    for column in self.plot_columns:
      kwargs = {
          'title': self.metadata.get_title(column),
          'x_label': x_label,
          'y_label': self.metadata.get_y_label(column),
          'legend': self.metadata.get_legend(column),
      }
      dict_kwargs[column] = kwargs
    return dict_kwargs

  def _process_grouped_data(self, data_copy: pd.DataFrame) -> pd.DataFrame:
    """
    Process grouped data and pivot if needed.

    Parameters
    ----------
    `data_copy` : `pd.DataFrame`
        The data to be processed.

    Returns
    -------
    `pd.DataFrame`
        The processed data.
    """
    reindexed_df = data_copy.reset_index()
    reindexed_df = self.format_index(reindexed_df)
    data_copy = self.remove_index_cols(reindexed_df)
    if self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.INDEX_COLS] != self.metadata.metadata[
            viz_schema.MetaDataSchema.FRAME][
                viz_schema.MetaDataSchema.GROUPED_COLS]:
      data_copy = self.pivot_data(data_copy)
    return data_copy

  def format_index(self, dataf: pd.DataFrame) -> pd.DataFrame:
    """
    Format the index of the data based on the number of index columns in the metadata.

    Parameters
    ----------
    `dataf` : `pd.DataFrame`
        The data to be formatted.
    
    Returns
    -------
    `pd.DataFrame`
        The formatted data.
    """
    if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.INDEX_COLS]) > 1:
      dataf.index = self._adjust_index(dataf)
    else:
      index_col = self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
          viz_schema.MetaDataSchema.INDEX_COLS][0]
      dataf.index = dataf[index_col]
    return dataf

  def _adjust_index(self, time_data: pd.DataFrame) -> pd.Series:
    """
    Adjust index to be a continuous variable.
    
    Parameters
    ----------
    `time_data` : `pd.DataFrame`
        The data to be adjusted.
        
    Returns
    -------
    `pd.Series`
        The adjusted index.
    """
    return time_data['Day of week'] + (
        1 / (time_data['Half-hour'].max() + 1)) * time_data['Half-hour']

  def remove_index_cols(self, dataf: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the index columns from the data after they've been reset.

    Parameters
    ----------
    `dataf` : `pd.DataFrame`
        The data to be processed.

    Returns
    -------
    `pd.DataFrame`
        The processed data.
    """
    return dataf.drop(columns=self.metadata.metadata[
        viz_schema.MetaDataSchema.FRAME][viz_schema.MetaDataSchema.INDEX_COLS],
                      axis=1)

  def pivot_data(self, dataf: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the data based on the metadata grouped columns.

    Parameters
    ----------
    `dataf` : `pd.DataFrame`
        The data to be pivoted.

    Returns
    -------
    `pd.DataFrame`
        The pivoted data.
    """
    legend_cols = []
    no_legend = (len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS]) -
                 len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
                     viz_schema.MetaDataSchema.INDEX_COLS]))
    for i in range(no_legend):
      legend_col = self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
          viz_schema.MetaDataSchema.GROUPED_COLS][i]
      legend_cols.append(legend_col)
    value_columns = [col for col in dataf.columns if col != legend_cols]
    return dataf.pivot(columns=legend_cols, values=value_columns)
