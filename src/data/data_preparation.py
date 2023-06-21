from copy import deepcopy
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from src.data import viz_schema


def convert_data_types(data: np.ndarray | pd.DataFrame,
                       columns: list[str]) -> np.ndarray | ValueError:
  if isinstance(data, pd.DataFrame):
    return np.ndarray(data.values)
  elif isinstance(data, np.ndarray):
    return pd.DataFrame(data, columns=columns)
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def describe_data(data: np.ndarray | pd.DataFrame) -> None | ValueError:
  if isinstance(data, pd.DataFrame):
    return data.describe()
  elif isinstance(data, np.ndarray):
    return pd.DataFrame(data).describe()
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def retrieve_data(data: np.ndarray | pd.DataFrame) -> np.ndarray | ValueError:
  if isinstance(data, pd.DataFrame):
    return data.values
  elif isinstance(data, np.ndarray):
    return data
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


class DataPreparationProtocol(Protocol):

  def data_cleaner(self) -> np.ndarray | pd.DataFrame:
    ...


class OutlierRemover():
  """ Removes outliers from array/dataframe """

  def __init__(self, data) -> None:
    self.data = data

  def data_cleaner(self) -> np.ndarray | pd.DataFrame:
    """
    Remove outliers from the data.
    Parameters:
      Data: Either numpy array or pandas dataframe
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    data_copy = deepcopy(self.data)
    values = retrieve_data(data_copy)
    outliers = self.find_outliers(values)
    data_copy = data_copy.astype(float)
    data_copy[outliers] = np.nan
    return data_copy

  def find_outliers(self, values: np.ndarray) -> np.ndarray:
    """
    Find outliers in the data.
    """
    outliers = np.zeros_like(values, dtype=bool)
    for column_idx in range(values.shape[1]):
      column_data = values[:, column_idx]
      q1 = np.percentile(column_data, 25)
      q3 = np.percentile(column_data, 75)
      iqr = q3 - q1
      lower_bound = q1 - (1.5 * iqr)
      upper_bound = q3 + (1.5 * iqr)
      outliers[:, column_idx] = (column_data < lower_bound) | (column_data
                                                               > upper_bound)
    return outliers


class FillMissingData():
  """
  Fills missing values in array/dataframe.
  """

  def __init__(self, data) -> None:
    self.data = data

  def data_cleaner(self, func: str) -> np.ndarray | pd.DataFrame:
    """
    Fill missing values in the data.

    Parameters:
      Data: Either numpy array or pandas dataframe
      Func: 'fillna' or 'dropna' at the moment fill only does mean 
      but could be expanded.
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    if func == "dropna":
      data = self.dropna()
    elif func == "fillna":
      data = self.fillna()
    else:
      raise ValueError(
          "Invalid fill method. Please choose 'dropna' or 'fillna'.")

    return data

  def dropna(self) -> np.ndarray | pd.DataFrame:
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, pd.DataFrame):
      return data_copy.dropna()
    elif isinstance(data_copy, np.ndarray):
      return data_copy[~np.isnan(data_copy).any(axis=1)]
    else:
      raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)

  def fillna(self) -> np.ndarray | pd.DataFrame:
    data_copy = deepcopy(self.data)
    if isinstance(data_copy, pd.DataFrame):
      return data_copy.fillna(data_copy.mean())
    elif isinstance(data_copy, np.ndarray):
      for column_idx in range(data_copy.shape[1]):
        data_copy[:, column_idx][np.isnan(
            data_copy[:, column_idx])] = np.nanmean(data_copy[:, column_idx],
                                                    axis=0)
      return data_copy
    else:
      raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


class GenerateDatetime():
  """
  Creates a datetime for dataset or without one.
  """

  def __init__(self, data) -> None:
    self.data = data.astype(float)

  def data_cleaner(self,
                   start_date: datetime = datetime(2022, 1, 1),
                   freq: str = "30T",
                   periods: int = 48) -> np.ndarray | pd.DataFrame:
    """
    Parameters:
      Data: Either numpy array or pandas dataframe
      Start_date: When will your datetime series start.
      freq: pd.Date_range variable defult to 30 minutes.
      periods: If no data is given length used to set the length 
               of array. Defult 48 giving 1 day with defult frequency.
    Returns:
      Input data with datetime column, index if Dataframe, column zero if array -> Needs to be converted to datetime.
    """
    data_copy = deepcopy(self.data)
    if self.data is not None:
      num_steps = len(self.data)
    else:
      num_steps = periods

    datetime_array = pd.date_range(start=start_date,
                                   periods=num_steps,
                                   freq=freq)
    if isinstance(self.data, np.ndarray):
      df = pd.DataFrame(index=datetime_array)
      return np.insert(data_copy, 0, df.index, axis=1)
    elif isinstance(self.data, pd.DataFrame):
      return data_copy.set_index(datetime_array)
