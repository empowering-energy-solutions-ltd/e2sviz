from copy import deepcopy
from typing import Protocol

import numpy as np
import pandas as pd
from e2slib.structures import datetime_schema, enums
from e2slib.utillib import functions

from src.data import viz_schema

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


class EquationManipulator():
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
      new_col = eval(f'data_copy[:,target_col] {func}')
      return np.insert(data_copy, data_copy.shape[1], new_col, axis=1)
    else:
      data_copy[new_col] = data_copy[target_col].apply(
          lambda x: eval(f'x {func}'))
      return data_copy


class CombineColumnManipulator():
  """
  Combine given columns and return as a new dataframe or array.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame, col_1: str | int,
               col_2: str | int) -> None:
    self.data = data
    self.col_1 = col_1
    self.col_2 = col_2

  def data_formatter(self) -> np.ndarray | pd.DataFrame:
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, np.ndarray):
      return np.insert(data_copy,
                       data_copy.shape[1],
                       data_copy[:, self.col_1] + data_copy[:, self.col_2],
                       axis=1)
    else:
      data_copy[viz_schema.ManipulationSchema.
                NEW_COL] = data_copy[self.col_1] + data_copy[self.col_2]
      return data_copy


class SeasonalWeekManipulator():
  """
  Creates a new dataframe/array of the data for 1 week of each season.
  Datetime column/index must be in the data. For array, it must be the first column.
  """

  def __init__(self, data: np.ndarray | pd.DataFrame,
               datetime_col: int | str | None) -> None:
    self.data = data
    self.datetime_col = datetime_col

  def datetime_check(self) -> None:
    """
    Checks if datetime column/index is in the data.
    """
    if isinstance(self.data, np.ndarray):
      if self.datetime_col is None:
        raise ValueError(
            'Datetime column/index must be specified for array data.')
    else:
      if isinstance(self.data, pd.DataFrame):
        index = self.data.index
        if not isinstance(index, pd.DatetimeIndex):
          raise ValueError(
              'Index must be pd.DatetimeIndex for dataframe data.')

  def data_formatter(self) -> list[np.ndarray] | list[pd.DataFrame]:
    self.datetime_check()
    data_copy = deepcopy(self.data)
    is_array = False
    if isinstance(self.data, np.ndarray):
      is_array = True
      data_copy = pd.DataFrame(self.data[:, 1:],
                               index=pd.DatetimeIndex(
                                   self.data[:, self.datetime_col]))
    data_copy = functions.add_time_features(data_copy)
    output = []
    frames = []
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
    for seasons in data_copy['season'].unique():
      found_week = False
      week_counter = 0
      for week in data_copy['Week'][data_copy['season'] == seasons].unique():
        if len(data_copy[(data_copy['season'] == seasons)
                         & (data_copy['Week'] == week)]) == 336:
          week_counter += 1
          if week_counter == 3:
            frames = data_copy[(data_copy['season'] == seasons)
                               & (data_copy['Week'] == week)]
            found_week = True  # Set the flag to True
            break  # Stop searching for additional weeks

      if not found_week:
        output.append(None)
      else:
        output.append(frames)
    if is_array:
      return np.array(output)
    return output
