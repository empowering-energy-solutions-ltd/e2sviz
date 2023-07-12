import datetime
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol

import numpy as np
import pandas as pd

from src.data import enums as viz_enums
from src.data import viz_schema


class LibraryViz(Protocol):
  """
  Selects the visualisation library to be used.
  """

  def plot_single(self, x: pd.DatetimeIndex, y: pd.Series, title: str,
                  x_label: str, y_label: str):
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

  def freq(self, col: str) -> viz_schema.FrequencySchema:
    ...

  def dtype(self, col: str) -> viz_enums.DataType:
    ...

  def get_x_label(self, col: str) -> str:
    ...

  def get_y_label(self, col: str) -> str:
    ...

  def get_title(self, col: str) -> str:
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
  meta_data : MetaData
    The meta data of the data to be visualised.

  Methods
  -------
  plot()
    Plots the data.
  
  """
  data: pd.DataFrame
  meta_data: MetaData
  viz_selector: LibraryViz

  def single_line_plot(self, cols: list[str]):
    """
    Plots the data.

    Parameters
    ----------
    cols : list[str], optional
      The columns to be plotted. If None, all columns are plotted.
    
    """
    for c in cols:
      if len(self.meta_data.metadata[c]['groupby_cols']) == 0:
        self.viz_selector.plot_single(x=self.data.index,
                                      y=self.data[c],
                                      title=self.meta_data.get_title(c),
                                      x_label=self.meta_data.get_x_label(c),
                                      y_label=self.meta_data.get_y_label(c))
      else:
        self.grouped_single_line_plot(c)
    pass

  def grouped_single_line_plot(self, col: str):
    reset_data = self.data.reset_index()
    index_cols = self.meta_data.metadata[col][
        viz_schema.MetaDataSchema.INDEX_COLS]
    if index_cols == ['Day of week', 'Half-hour']:
      reset_data['index_int'] = self.day_and_time(reset_data)
    legends = self.meta_data.metadata[col][viz_schema.MetaDataSchema.LEGEND]

    if len(legends) > 0:
      grouped = reset_data.groupby(legends)
      for name, group in grouped:
        if index_cols != ['Day of week', 'Half-hour']:
          self.viz_selector.plot_single(
              x=group[index_cols],
              y=group[col],
              title=self.meta_data.get_title(col),
              x_label=index_cols[0],
              y_label=self.meta_data.get_y_label(col))
        else:
          self.viz_selector.plot_single(
              x=group['index_int'],
              y=group[col],
              title=self.meta_data.get_title(col),
              x_label=index_cols[0],
              y_label=self.meta_data.get_y_label(col))
    else:
      pass

  def day_and_time(self, time_data) -> pd.Series:
    return time_data['Day of week'] + (
        1 / time_data['Half-hour'].max()) * time_data['Half-hour']

  def multi_plot(self):
    pass

  def bar_box_plot(self):
    pass

  def scatter_plot(self):
    pass

  def correlation_plot(self):
    pass
