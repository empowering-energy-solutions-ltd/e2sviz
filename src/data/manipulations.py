from copy import deepcopy
from typing import Protocol

import numpy as np
import numpy.typing as npt
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


class ResampleManipulator():
  """
  Returns resampled data.
  """

  def data_formatter(self, data: npt.NDArray | pd.DataFrame, split_by: str,
                     aggregation: str) -> npt.NDArray | pd.DataFrame:
    """
    Takes all data and returns just for specified year
    Parameters:
      Data: Either numpy array or pandas dataframe
      split_by: str, resample method
      aggregation: str, aggregation method
    Returns:
      Array or Dataframe, whichever you gave it in the first place resampled.
    """
    data_copy = deepcopy(data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      return data_copy.resample(split_by).agg(aggregation).to_numpy()
    else:
      return data_copy.resample(split_by).agg(aggregation)


class AddTimeFeatureManipulator():
  """
  Adds the e2slib time features to the data.
  """

  def data_formatter(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """ 
    Takes data and applies the e2slib time features to it.
    Parameters:
      Data: Either numpy array or pandas dataframe
    Returns:
      Array or Dataframe, whichever you gave it in the first place with time features added.
    """
    data_copy = deepcopy(data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      arr_with_time = functions.add_time_features(data_copy).to_numpy()
      return np.insert(arr_with_time, 0, data_copy.index, axis=1)
    else:
      return functions.add_time_features(data_copy)


class GroupbyManipulator():
  """
  Returns data grouped by choosen column.
  """

  def data_formatter(
      self,
      data: npt.NDArray | pd.DataFrame,
      groupby: list[int | str],
      agg: str,
      target: int | str | None = None) -> npt.NDArray | pd.DataFrame:
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
    data_copy = deepcopy(data)
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

  def data_formatter(
      self,
      data: npt.NDArray | pd.DataFrame,
      target_col: str | int,
      func: str,
      new_col: str | float = 'New column') -> npt.NDArray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """
    data_copy = deepcopy(data)
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

  def data_formatter(self, data: npt.NDArray | pd.DataFrame, col_1: str | int,
                     col_2: str | int) -> npt.NDArray | pd.DataFrame:
    data_copy = deepcopy(data)
    if isinstance(data_copy, np.ndarray):
      return np.insert(data_copy,
                       data_copy.shape[1],
                       data_copy[:, col_1] + data_copy[:, col_2],
                       axis=1)
    else:
      data_copy[viz_schema.ManipulationSchema.
                NEW_COL] = data_copy[col_1] + data_copy[col_2]
      return data_copy


class SeasonalWeekManipulator():
  """
  Creates a new dataframe/array of the data for 1 week of each season.
  Datetime column/index must be in the data. For array, it must be the first column.
  """

  def data_formatter(
      self, data: npt.NDArray | pd.DataFrame,
      datetime_col: int | str | None) -> list[np.ndarray] | list[pd.DataFrame]:
    self.datetime_check(data, datetime_col)
    data_copy = deepcopy(data)
    is_array = False
    if isinstance(data, np.ndarray):
      is_array = True
      data_copy = pd.DataFrame(data[:, 1:],
                               index=pd.DatetimeIndex(data[:, datetime_col]))
    # if isinstance(data_copy, np.ndarray):
    #   data_copy = pd.DataFrame(data_copy[:, 1:],
    #                            index=pd.DatetimeIndex(data_copy[:, 0]))

    loop_data = functions.add_time_features(data_copy)
    output = []
    frames = []
    for seasons in loop_data['season'].unique():
      found_week = False
      week_counter = 0
      for week in loop_data['Week'][loop_data['season'] == seasons].unique():
        if len(loop_data[(loop_data['season'] == seasons)
                         & (loop_data['Week'] == week)]) == 336:
          week_counter += 1
          if week_counter == 3:
            frames = loop_data[(loop_data['season'] == seasons)
                               & (loop_data['Week'] == week)]
            found_week = True  # Set the flag to True
            break  # Stop searching for additional weeks

      if not found_week:
        output.append(None)
      else:
        output.append(frames)
    if is_array:
      return np.array(output)
    return output

  def datetime_check(self, data: npt.NDArray | pd.DataFrame,
                     datetime_col: int | str | None) -> None:
    """
    Checks if datetime column/index is in the data.
    """
    if isinstance(data, np.ndarray):
      if datetime_col is None:
        raise ValueError(
            'Datetime column/index must be specified for array data.')
    else:
      if isinstance(data, pd.DataFrame):
        index = data.index
        if not isinstance(index, pd.DatetimeIndex):
          raise ValueError(
              'Index must be pd.DatetimeIndex for dataframe data.')
