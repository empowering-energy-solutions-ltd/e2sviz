import datetime
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol

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
      reindex_df: pd.DataFrame = dataf.reset_index()
      if len(self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
          viz_schema.MetaDataSchema.INDEX_COLS]) > 1:

        reindex_df.index = self.day_and_time(reindex_df)
        dataf = reindex_df.drop(
            columns=self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
                viz_schema.MetaDataSchema.INDEX_COLS],
            axis=1)
      else:
        index_col = self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
            viz_schema.MetaDataSchema.INDEX_COLS][0]
        reindex_df.index = reindex_df[index_col]
        dataf = reindex_df.drop(
            columns=self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
                viz_schema.MetaDataSchema.INDEX_COLS],
            axis=1)
      if self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
          viz_schema.MetaDataSchema.INDEX_COLS] != self.metadata.metadata[
              viz_schema.MetaDataSchema.FRAME][
                  viz_schema.MetaDataSchema.GROUPED_COLS]:
        legend_col = self.metadata.metadata[viz_schema.MetaDataSchema.FRAME][
            viz_schema.MetaDataSchema.GROUPED_COLS][0]
        value_columns = [col for col in dataf.columns if col != legend_col]
        dataf = dataf.pivot(columns=legend_col, values=value_columns)
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
      return self.viz_selector.plot_single(x=dataf.index,
                                           y=dataf[c],
                                           kwargs=kwargs)  #.show()

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
      kwargs['title'] = 'Barplot of column sums'
      return self.viz_selector.bar_plot(data=dataf, kwargs=kwargs)  #.show()
    else:
      kwargs['title'] = 'Boxplot of columns'
      return self.viz_selector.box_plot(data=dataf, kwargs=kwargs)  #.show()

  def pie_chart_plot(self):
    dataf = self.data.copy()
    kwargs = {
        'title': 'Piechart of column sums',
        'x_label': 'Columns',
        'y_label': 'Column Values',
        'legend': [],
    }
    return self.viz_selector.pie_chart(data=dataf, kwargs=kwargs)  #.show()

  def scatter_plot(self):
    pass

  def correlation_plot(self):

    corr_matrix = self.data.corr()
    return self.viz_selector.corr_plot(corr_matrix)  #.show()
