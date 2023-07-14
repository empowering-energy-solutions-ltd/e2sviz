import datetime
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data import enums as viz_enums
from src.data import viz_schema


class LibraryViz(Protocol):
  """
  Selects the visualisation library to be used.
  """

  def plot_single(self, x: pd.DatetimeIndex | pd.Series, y: pd.Series,
                  kwargs) -> plt.Axes | go.Figure:
    ...

  def corr_plot(self, corr_matrix) -> plt.Figure | go.Figure:
    ...

  def bar_plot(self, data: pd.DataFrame,
               kwargs: dict[str, str]) -> plt.Figure | go.Figure:
    ...

  def box_plot(self, data: pd.DataFrame,
               kwargs: dict[str, str]) -> plt.Figure | go.Figure:
    ...

  def pie_chart(self, data: pd.Series,
                kwargs: dict[str, str]) -> plt.Figure | go.Figure:
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


class DataManipProtocol(Protocol):
  """
  Manipulates the data.
  """
  data: pd.DataFrame
  frequency: viz_schema.FrequencySchema
  column_meta_data: MetaData

  def __post_init__(self) -> None:
    ...

  def check_freq(self) -> None:
    ...

  def check_meta_data(self) -> None:
    ...

  def check_rescaling(self) -> None:
    ...

  @property
  def column_from_freq(self) -> str:
    ...

  @property
  def dict_of_groupbys(self) -> dict[str, List[str]]:
    ...

  def filter(self,
             year: Optional[List[int]] = None,
             month: Optional[List[int]] = None,
             day: Optional[List[int]] = None,
             hour: Optional[List[int]] = None,
             date: Optional[List[datetime.date]] = None,
             inplace: bool = False) -> Optional[pd.DataFrame]:
    ...

  def groupby(self,
              groupby_type: str = 'week_season',
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> pd.DataFrame | pd.Series:
    ...

  def resample(self,
               freq: str = 'D',
               func: Callable[[pd.DataFrame], pd.Series] = np.mean,
               inplace: bool = False) -> pd.DataFrame | pd.Series:
    ...

  def rolling(self,
              window: int = 3,
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> pd.DataFrame | pd.Series:
    ...


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

  def single_line_plot(self, cols: list[str] | None = None):
    """
    Plots the data.

    Parameters
    ----------
    cols : list[str], optional
      The columns to be plotted. If None, all columns are plotted.
    
    """
    dataf: pd.DataFrame = self.data.copy()
    if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS]) > 0:
      dataf.index = self.day_and_time(dataf.reset_index())
      print(dataf)
    if cols is None:
      cols: list[str] = dataf.columns
    for c in cols:
      # for legend in self.metadata.metadata[c][
      #     viz_schema.MetaDataSchema.LEGEND]:
      kwargs = {
          'title': self.metadata.get_title(c),
          'x_label': self.metadata.get_x_label,
          'y_label': self.metadata.get_y_label(c),
          'legend': self.metadata.get_legend(c),
      }
      self.viz_selector.plot_single(x=dataf.index, y=dataf[c],
                                    kwargs=kwargs).show()

  def day_and_time(self, time_data) -> pd.Series:
    """Adjust the index based on the column data"""
    return time_data['Day of week'] + (
        1 / (time_data['Half-hour'].max() + 1)) * time_data['Half-hour']

  def multi_plot(self):
    pass

  def bar_box_plot(self, bar: bool = True):
    dataf = self.data.copy()
    kwargs = {
        'title': 'Barplot of column sums',
        'x_label': 'Columns',
        'y_label': 'Column Values',
        'legend': [],
    }
    if bar:
      self.viz_selector.bar_plot(data=dataf, kwargs=kwargs).show()
    else:
      self.viz_selector.box_plot(data=dataf, kwargs=kwargs).show()

  def pie_chart_plot(self):
    dataf = self.data.copy()
    kwargs = {
        'title': 'Barplot of column sums',
        'x_label': 'Columns',
        'y_label': 'Column Values',
        'legend': [],
    }
    self.viz_selector.pie_chart(data=dataf, kwargs=kwargs).show()

  def scatter_plot(self):
    pass

  def correlation_plot(self):

    corr_matrix = self.data.corr()
    self.viz_selector.corr_plot(corr_matrix).show()

    #   # reindex_dataframe(dataf)
    #   # groupby_title(self.metadata)

    #   plot_data = self.grouped_single_line_plot(c)
    # else:
    #   kwargs = {
    #       'title': self.metadata.get_title(c),
    #       'x_label': self.metadata.get_x_label(c),
    #       'y_label': self.metadata.get_y_label(c),
    #       'legend': self.metadata.get_legend(c)
    #   }
    #   plot_data = [(dataf, kwargs)]
    # for dataf, kwargs in plot_data:

    #   self.viz_selector.plot_single(x=dataf.index, y=dataf[c], kwargs=kwargs)

  # def grouped_single_line_plot(self, col: str) -> tuple[pd.DataFrame, dict]:
  #   """
  #   Generates individual plots for each element of the grouped data.

  #   Parameters
  #   ----------
  #   col : str
  #     The column to be plotted.

  #   """
  #   plot_data = []
  #   reordered_data = self.data.reset_index()
  #   index_cols = self.metadata.metadata[col][
  #       viz_schema.MetaDataSchema.INDEX_COLS]
  #   if index_cols == viz_schema.MetaDataSchema.DAY_HOUR:
  #     reordered_data.index = self.day_and_time(reordered_data)
  #   legends = self.metadata.metadata[col][viz_schema.MetaDataSchema.LEGEND]

  #   if len(legends) > 0:
  #     grouped = reordered_data.groupby(legends)
  #     for name, group in grouped:
  #       kwargs = {
  #           'title': self.metadata.get_title(col,
  #                                            name[0]),  #f'{} - {name[0]}',
  #           'x_label': index_cols[0],
  #           'y_label': self.metadata.get_y_label(col),
  #           'legend': name[0]
  #       }
  #       plot_data.append((group, kwargs))
  #     return plot_data
  #   else:
  #     kwargs = {
  #         'title': self.metadata.get_title(col),
  #         'x_label': index_cols[0],
  #         'y_label': self.metadata.get_y_label(col),
  #         'legend': []
  #     }
  #     plot_data.append((reordered_data, kwargs))
  #     return plot_data