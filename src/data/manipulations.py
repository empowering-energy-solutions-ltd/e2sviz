from copy import deepcopy
from typing import Protocol

import numpy as np
import pandas as pd
from e2slib.structures import datetime_schema, enums
from e2slib.utillib import functions

# from data_preperation import retrieve_data


def create_seasonal_average_week(season: enums.Season,
                                 dataf: pd.DataFrame,
                                 target_col: str | None = None,
                                 func=np.mean) -> pd.DataFrame:
  timeseries_data = functions.add_time_features(dataf).copy()
  filt = timeseries_data[datetime_schema.DateTimeSchema.SEASON] == season.name
  cols = [
      datetime_schema.DateTimeSchema.DAYOFWEEK,
      datetime_schema.DateTimeSchema.HALFHOUR
  ]
  if target_col is None:
    target = timeseries_data.columns[0]
    seasonal_data = timeseries_data[filt].groupby(cols).agg({target: func})
  else:
    seasonal_data = timeseries_data[filt].groupby(cols).agg({target_col: func})
  new_index = functions.format_avg_week_index(seasonal_data,
                                              enums.TimeStep.HALFHOUR)
  seasonal_data.index = new_index
  return seasonal_data


class DataFormattingProtocol(Protocol):
  """
  Formats the data in different ways to allow for various visualisations.
  Can be skipped for plotting full datasets.
  """

  def data_formatter(self) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """


class ResampleManipulator(DataFormattingProtocol):
  """
  Returns resampled data.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame) -> None:
    self.data = data

  def data_formatter(self, split_by: str,
                     aggregation: str) -> np.ndarray | pd.DataFrame:
    """
    Takes all data and returns just for specified year
    Parameters:
      Data: Either numpy array or pandas dataframe
      split_by: str, resample method
      aggregation: str, aggregation method
    Returns:
      Array or Dataframe, whichever you gave it in the first place resampled.
    """
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      return data_copy.resample(split_by).agg(aggregation).to_numpy()
    else:
      return data_copy.resample(split_by).agg(aggregation)


class AddTimeFeatureManipulator(DataFormattingProtocol):
  """
  Adds the e2slib time features to the data.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame) -> None:
    self.data = data

  def data_formatter(self) -> np.ndarray | pd.DataFrame:
    """ 
    Takes data and applies the e2slib time features to it.
    Parameters:
      Data: Either numpy array or pandas dataframe
    Returns:
      Array or Dataframe, whichever you gave it in the first place with time features added.
    """
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      arr_with_time = functions.add_time_features(data_copy).to_numpy()
      return np.insert(arr_with_time, 0, data_copy.index, axis=1)
    else:
      return functions.add_time_features(data_copy)


class GroupbyManipulator(DataFormattingProtocol):
  """
  Returns data grouped by choosen column.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame) -> None:
    self.data = data

  def data_formatter(
      self,
      groupby: list[int | str],
      agg: str,
      target: int | str | None = None) -> np.ndarray | pd.DataFrame:
    """ 
    Groups data by choosen column.
    Parameters:
      Data: Either numpy array or pandas dataframe
      target: str, column to aggregate
      groupby: list, columns to group by
      agg: str, aggregation method
    Returns:
      Array or Dataframe, whichever you gave it in the first place grouped by target.
    """
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      return data_copy.groupby(groupby).agg({target: agg}).to_numpy()
    else:
      return data_copy.groupby(groupby).agg({target: agg})


class EquationManipulator(DataFormattingProtocol):
  """
  Returns data with new column of some aggregation.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame) -> None:
    self.data = data

  def data_formatter(
      self,
      target_col: str | int,
      func: str,
      new_col: str | float = 'New column') -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, np.ndarray):
      # data_copy = pd.DataFrame(data_copy[:, 1:],
      #                          index=pd.DatetimeIndex(data_copy[:, 0]))
      new_col = eval(f'data_copy[:,target_col] {func}')
      return np.insert(data_copy, data_copy.shape[1], new_col, axis=1)
    else:
      data_copy[new_col] = data_copy[target_col].apply(
          lambda x: eval(f'x {func}'))
      return data_copy
