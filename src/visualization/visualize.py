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

  def column_name(self, col: str) -> viz_enums.DataType:
    ...

  def get_x_label(self, col: str) -> str:
    ...

  def get_y_label(self, col: str) -> str:
    ...

  def get_title(self, col: str) -> str:
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
  datamanip: DataManipProtocol
  viz_selector: LibraryViz

  @property
  def data(self) -> pd.DataFrame:
    return self.datamanip.data

  @property
  def meta_data(self) -> MetaData:
    return self.datamanip.column_meta_data

  def single_plot(self, cols: Optional[list[str]] = None):
    if 'Indexes' in self.meta_data.metadata:
      print('Indexes found.  No plotting available.')
    else:
      if cols:
        for c in cols:
          self.viz_selector.plot_single(x=self.data.index,
                                        y=self.data[c],
                                        title=self.meta_data.get_title(c),
                                        x_label=self.meta_data.get_x_label(c),
                                        y_label=self.meta_data.get_y_label(c))
      else:
        for c in self.data.columns:
          self.viz_selector.plot_single(x=self.data.index,
                                        y=self.data[c],
                                        title=self.meta_data.get_title(c),
                                        x_label=self.meta_data.get_x_label(c),
                                        y_label=self.meta_data.get_y_label(c))

  def multi_plot(self):
    pass

  def bar_box_plot(self):
    pass

  def scatter_plot(self):
    pass

  def correlation_plot(self):
    pass
