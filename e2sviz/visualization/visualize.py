import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from e2sviz.structure import enums as viz_enums
from e2sviz.structure import viz_schema


class LibraryViz(Protocol):
  """
  Selects the visualisation library to be used.
  """

  def plot_single(self, dataf: pd.DataFrame, plot_columns: list[str],
                  dict_kwargs: dict[str, dict[str, str]]):
    ...

  def corr_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    ...

  def bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
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
  Stores the meta data.
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


# # class DataManipProtocol(Protocol):

#   """
#   Manipulates the data.
#   """
#   data: pd.DataFrame
#   frequency: viz_schema.FrequencySchema
#   column_meta_data: MetaData

#   def __post_init__(self) -> None:
#     ...

#   def check_freq(self) -> None:
#     ...

#   def check_meta_data(self) -> None:
#     ...

#   def check_rescaling(self) -> None:
#     ...

#   @property
#   def column_from_freq(self) -> str:
#     ...

#   @property
#   def dict_of_groupbys(self) -> dict[str, List[str]]:
#     ...

#   def filter(self,
#              year: Optional[List[int]] = None,
#              month: Optional[List[int]] = None,
#              day: Optional[List[int]] = None,
#              hour: Optional[List[int]] = None,
#              date: Optional[List[datetime.date]] = None,
#              inplace: bool = False) -> Self:
#     ...

#   def groupby(self,
#               groupby_type: str = 'week_season',
#               func: Callable[[pd.DataFrame], pd.Series] = np.mean,
#               inplace: bool = False) -> Self:
#     ...

#   def resample(self,
#                freq: str = 'D',
#                func: Callable[[pd.DataFrame], pd.Series] = np.mean,
#                inplace: bool = False) -> Self:
#     ...

#   def rolling(self,
#               window: int = 3,
#               func: Callable[[pd.DataFrame], pd.Series] = np.mean,
#               inplace: bool = False) -> Self:
#     ...


@dataclass
class DataViz:
  """
  Visualises the data.

  Parameters
  ----------
  DataManip : DataManipProtocol
    The data manipulation class.
  viz_selector : LibraryViz
    The visualisation library to be used.
  
  Attributes
  ----------
  data : pd.DataFrame
    The data to be visualised.
  metadata : MetaData
    The meta data of the data to be visualised.

  Methods
  -------
  plot()
    Plots the data.
  
  """
  data: pd.DataFrame
  metadata: MetaData
  viz_selector: LibraryViz
  plot_columns: Optional[list[str]] = None

  def __post_init__(self):
    if self.plot_columns is None:
      self.plot_columns: list[str] = self.data.columns.to_list()

  @property
  def plot_factory(self) -> dict[str, Callable]:
    return {
        'single_plot': self.viz_selector.plot_single,
        'corr_plot': self.viz_selector.corr_plot,
        'bar_plot': self.viz_selector.bar_plot,
        'box_plot': self.viz_selector.box_plot,
        'pie_chart': self.viz_selector.pie_chart
    }

  def plot(self, plot_kind):
    plot_data = self.structured_data()
    dict_kwargs = self.create_dict_kwargs()
    self.plot_factory[plot_kind](dataf=plot_data,
                                 plot_columns=self.plot_columns,
                                 dict_kwargs=dict_kwargs)
    # return self.viz_selector.show()

  def show_viz(self):
    self.viz_selector.show()

  def save_figure(self):
    self.viz_selector.save()

  def structured_data(self) -> pd.DataFrame:
    """
    Returns the data in a structured format.
    
    Returns
    -------
    pd.DataFrame
      The data in a structured format.
    """
    data_copy = self.data.copy()
    if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS]) > 0:
      data_copy = self._process_grouped_data(data_copy)
    return data_copy

  def create_dict_kwargs(self) -> dict[str, dict[str, Any]]:
    """
    Creates the list of kwargs for each column.

    Returns
    -------
    list[dict[str, Any]]
      The list of kwargs for each column.
    """
    dict_kwargs = {}
    for column in self.plot_columns:
      kwargs = {
          'title': self.metadata.get_title(column),
          'x_label': self.metadata.get_x_label,
          'y_label': self.metadata.get_y_label(column),
          'legend': self.metadata.get_legend(column),
      }
      dict_kwargs[column] = kwargs
    return dict_kwargs

  def _process_grouped_data(self, data_copy: pd.DataFrame) -> pd.DataFrame:
    """Process grouped data and pivot if needed."""
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
    dataf : pd.DataFrame
        The data to be formatted.
    
    Returns
    -------
    pd.DataFrame
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
    time_data : pd.DataFrame
        The data to be adjusted.
        
    Returns
    -------
    pd.Series
        The adjusted index.
    """
    return time_data['Day of week'] + (
        1 / (time_data['Half-hour'].max() + 1)) * time_data['Half-hour']

  def remove_index_cols(self, dataf: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the index columns from the data after they've been reset.
    """
    return dataf.drop(columns=self.metadata.metadata[
        viz_schema.MetaDataSchema.FRAME][viz_schema.MetaDataSchema.INDEX_COLS],
                      axis=1)

  def pivot_data(self, dataf: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the data based on the metadata grouped columns.
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

    # def single_line_plot(
    #     self,
    #     cols: Optional[List[str]] = None,
    #     fig_ax: Optional[go.Figure | plt.Axes] = None) -> plt.Axes | go.Figure:
    #   """
    #       Plots the data.

    #       Parameters
    #       ----------
    #       cols : Optional[List[str]], optional
    #           The columns to be plotted. If None, all columns are plotted.

    #       Returns
    #       -------
    #       plt.Axes | go.Figure
    #           The plot.
    #       """
    #   data_copy = self.data.copy()
    #   if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
    #       viz_schema.MetaDataSchema.GROUPED_COLS]) > 0:
    #     data_copy = self._process_grouped_data(data_copy)
    #   if cols is None:
    #     cols = data_copy.columns
    #   for col in cols:
    #     kwargs = {
    #         'title': self.metadata.get_title(col),
    #         'x_label': self.metadata.get_x_label,
    #         'y_label': self.metadata.get_y_label(col),
    #         'legend': self.metadata.get_legend(col),
    #     }
    #     return self.viz_selector.plot_single(x=data_copy.index,
    #                                          y=data_copy[col],
    #                                          fig_ax=fig_ax,
    #                                          kwargs=kwargs)

    # def multi_plot(self):
    #   pass

    # def bar_plot(self,
    #              cols: Optional[list[str]] = None,
    #              sum_vals: bool = False) -> plt.Axes | go.Figure:
    #   """
    #   Plots a barplot of the columns.

    #   Parameters
    #   ----------
    #   cols : Optional[list[str]], optional
    #       The columns to be plotted. If None, all columns are plotted.
    #   sum : bool, optional
    #       If True, sums the columns. If False, plots the columns as is. The default is False.

    #   Returns
    #   -------
    #   plt.Axes | go.Figure
    #       Barplot or boxplot of column data.
    #   """
    #   if cols is None:
    #     cols: list[str] = self.data.columns
    #   kwargs: dict[str, Any] = {
    #       'title': 'Barplot of columns',
    #       'x_label': 'Columns',
    #       'y_label': 'Energy (kWh)',
    #       'legend': cols,
    #   }
    #   return self.viz_selector.bar_plot(data=self.data,
    #                                     kwargs=kwargs,
    #                                     cols=cols,
    #                                     sum_vals=sum_vals)

    # def box_plot(self) -> plt.Axes | go.Figure:
    #   """
    #   Plots a boxplot of the columns.

    #   Parameters
    #   ----------
    #   bar : bool, optional
    #       If True, plots a barplot. If False, plots a boxplot. The default is True.

    #   Returns
    #   -------
    #   plt.Axes | go.Figure
    #       Barplot or boxplot of column data.
    #   """
    #   dataf = self.data.copy()
    #   kwargs = {
    #       'title': 'Boxplot of columns',
    #       'x_label': 'Columns',
    #       'y_label': 'Energy (kWh)',
    #       'legend': [],
    #   }
    #   return self.viz_selector.box_plot(data=dataf, kwargs=kwargs)  #.show()

    # def pie_chart_plot(self) -> plt.Axes | go.Figure:
    #   """
    #   Plots a pie chart of the column sums.

    #   Returns
    #   -------
    #   plt.Axes | go.Figure
    #       Pie chart plot."""
    #   dataf = self.data.copy()
    #   kwargs = {
    #       'title': 'Piechart of column sums',
    #       'x_label': 'Columns',
    #       'y_label': 'Column Values',
    #       'legend': [],
    #   }
    #   return self.viz_selector.pie_chart(data=dataf, kwargs=kwargs)  #.show()

    # def scatter_plot(self):
    #   pass

    # def correlation_plot(self) -> plt.Axes | go.Figure:
    """
    Plots the correlation matrix of the data.

    Returns
    -------
    plt.Axes | go.Figure
        The correlation plot.
    """
    corr_matrix = self.data.corr()
    return self.viz_selector.corr_plot(corr_matrix)  #.show()