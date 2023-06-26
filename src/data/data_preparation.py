import warnings
from copy import deepcopy
from datetime import datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data import viz_schema


def convert_data_types(
    data: npt.NDArray | pd.DataFrame,
    columns: list[str] | None = None) -> np.ndarray | ValueError:
  if isinstance(data, pd.DataFrame):
    return np.ndarray(data.values)
  elif isinstance(data, np.ndarray):
    if columns:
      return pd.DataFrame(data)
    else:
      return pd.DataFrame(data, columns=columns)
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def describe_data(data: npt.NDArray | pd.DataFrame) -> None | ValueError:
  if isinstance(data, pd.DataFrame):
    return data.describe()
  elif isinstance(data, np.ndarray):
    return pd.DataFrame(data).describe()
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def retrieve_data(data: npt.NDArray | pd.DataFrame) -> np.ndarray | ValueError:
  if isinstance(data, pd.DataFrame):
    return data.values
  elif isinstance(data, np.ndarray):
    return data
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def check_dataset(data: npt.NDArray | pd.DataFrame) -> dict[str, bool]:
  """
  Check the dataset for outliers and NaN values.
  """
  result = {'outliers': False, 'nan values': False, 'timeseries': False}
  result['timeseries'] = check_datetime(data)
  if isinstance(data, np.ndarray):
    # Check for outliers
    outliers_mask = np.abs(data - np.mean(data)) > 3 * np.std(data)
    result['outliers'] = np.any(outliers_mask)
    # Check for NaN values
    result['nan values'] = np.isnan(data).any()
  elif isinstance(data, pd.DataFrame):
    # Check for outliers
    outliers_mask = np.abs(data - data.mean()) > 3 * data.std()
    result['outliers'] = np.any(outliers_mask.values)
    # Check for NaN values
    result['nan values'] = data.isnull().values.any()
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)
  if result['outliers'] == True:
    result['nan values'] = True
  return result


def check_datetime(data: pd.DataFrame | npt.NDArray) -> bool:
  """
    Check if a datetime index or column is present in the given data.
    """
  if isinstance(data, pd.DataFrame):
    return _check_dataframe(data)
  elif isinstance(data, np.ndarray):
    return _check_ndarray(data)
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def _check_dataframe(data: pd.DataFrame) -> bool:
  """Check if a datetime index or column is present in the pandas DataFrame."""
  index_type = data.index.dtype
  if pd.api.types.is_datetime64_dtype(index_type):
    warnings.warn(
        'Warning. Your datetime index is not timezone aware. Recommend converting using tz_localize.'
    )
    return True
  if pd.api.types.is_datetime64tz_dtype(data.index.dtype):
    return True
  for column in data.columns:
    column_type = data[column].dtype
    if pd.api.types.is_datetime64_dtype(column_type):
      return True

  return False


def _check_ndarray(data: npt.NDArray) -> bool:
  """Check if a datetime column is present in the NumPy array."""
  # Convert the NumPy array to a pandas DataFrame for easier datetime detection
  df = pd.DataFrame(data)

  for column in df.columns:
    column_type = df[column].dtype
    if pd.api.types.is_datetime64_dtype(column_type):
      return True

  return False


class OutlierRemover():
  """ Removes outliers from array/dataframe """

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Remove outliers from the data.
    Parameters:
      Data: Either numpy array or pandas dataframe
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    data_copy = deepcopy(data)
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

  def data_cleaner(self, data: npt.NDArray | pd.DataFrame,
                   func: str) -> npt.NDArray | pd.DataFrame:
    """
    Fill missing values in the data.

    Parameters:
      Data: Either numpy array or pandas dataframe
      Func: 'fillna' or 'dropna' at the moment fill only does mean 
      but could be expanded.
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    if func == 'dropna':
      data = self.dropna(data)
    elif func == 'meanfill':
      data = self.fillna_mean(data)
    elif func == 'rollingfill':
      data = self.fillna_rolling(data)
    else:
      raise ValueError(
          "Invalid fill method. Please choose 'dropna', 'meanfill' or 'rollingfill'."
      )

    return data

  def dropna(self,
             data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    data_copy = deepcopy(data)
    if isinstance(data_copy, pd.DataFrame):
      return data_copy.dropna()
    elif isinstance(data_copy, np.ndarray):
      return data_copy[~np.isnan(data_copy).any(axis=1)]
    else:
      raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)

  def fillna_mean(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    data_copy = deepcopy(data)
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

  def fillna_rolling(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    data_copy = deepcopy(data)
    if isinstance(data_copy, pd.DataFrame):
      return data_copy.fillna(
          data_copy.rolling(window=3, min_periods=1, axis=0).mean())
    elif isinstance(data_copy, np.ndarray):
      for column_idx in range(data_copy.shape[1]):
        column_data = data_copy[:, column_idx]
        mask = np.isnan(column_data)
        rolling_mean = pd.Series(column_data).rolling(
            window=3, min_periods=1).mean().to_numpy()
        column_data[mask] = rolling_mean[mask]
      return data_copy
    else:
      raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


class GenerateDatetime():
  """
  Creates a datetime for dataset or without one.
  """

  def data_cleaner(self,
                   data: npt.NDArray | pd.DataFrame,
                   start_date: datetime = datetime(2022, 1, 1),
                   freq: str = "30T",
                   periods: int = 48,
                   tz: str = 'UTC') -> npt.NDArray | pd.DataFrame:
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
    data_copy = deepcopy(data)
    if data is not None:
      num_steps = len(data)
    else:
      num_steps = periods

    datetime_array = pd.date_range(start=start_date,
                                   periods=num_steps,
                                   freq=freq,
                                   tz=tz)
    if isinstance(data, np.ndarray):
      df = pd.DataFrame(index=datetime_array)
      return np.insert(data_copy, 0, df.index, axis=1)
    elif isinstance(data, pd.DataFrame):
      return data_copy.set_index(datetime_array)
