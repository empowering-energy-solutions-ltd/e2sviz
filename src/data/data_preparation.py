import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data import viz_schema


def convert_data_types(
    data: npt.NDArray | pd.DataFrame,
    columns: list[str] | None = None) -> np.ndarray | ValueError:
  """
  Convert data types between NumPy ndarray and pandas DataFrame.

  Parameters
  ----------
  data : np.ndarray or pd.DataFrame
      Input data to be converted.
  columns : List[str], None
      List of column names for DataFrame conversion. Defaults to None.

  Returns
  -------
  np.ndarray or pd.DataFrame
      Converted data in the specified type.

  Raises
  ------
  ValueError
      Raised when an unsupported data type is provided.
  """
  if isinstance(data, pd.DataFrame):
    return np.ndarray(data.values)
  elif isinstance(data, np.ndarray):
    if columns:
      return pd.DataFrame(data)  # type: ignore
    else:
      return pd.DataFrame(data, columns=columns)  # type: ignore
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def describe_data(
    data: npt.NDArray | pd.DataFrame) -> pd.DataFrame | ValueError:
  """
  Generate descriptive statistics for the input data.

  Parameters
  ----------
  data : np.ndarray or pd.DataFrame
      Input data for which to generate statistics.

  Returns
  -------
  None or pd.DataFrame
      DataFrame containing the descriptive statistics.

  Raises
  ------
  ValueError
      Raised when an unsupported data type is provided.
  """
  if isinstance(data, pd.DataFrame):
    return data.describe()
  elif isinstance(data, np.ndarray):
    return pd.DataFrame(data).describe()
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def retrieve_data(data: npt.NDArray | pd.DataFrame) -> np.ndarray | ValueError:
  """
  Retrieve the underlying NumPy ndarray from the input data.

  Parameters
  ----------
  data : np.ndarray or pd.DataFrame
      Input data from which to retrieve the ndarray.

  Returns
  -------
  np.ndarray
      NumPy ndarray representing the input data.

  Raises
  ------
  ValueError
      Raised when an unsupported data type is provided.
  """
  if isinstance(data, pd.DataFrame):
    return data.values
  elif isinstance(data, np.ndarray):
    return data
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def check_dataset(data: npt.NDArray | pd.DataFrame) -> dict[str, bool]:
  """
    Check the dataset for outliers and NaN values.

    Parameters
    ----------
    data : np.ndarray | pd.DataFrame
        Input data to be checked.
    columns : list[str] | None, optional
        Columns to consider when checking for outliers in an np.ndarray, by default None.

    Returns
    -------
    dict[str, bool]
        Dictionary containing the results of outlier and NaN checks.

    Raises
    ------
    ValueError
        Raised when an unsupported data type is provided.
  """
  result = {'outliers': False, 'nan values': False, 'timeseries': False}
  result['timeseries'] = check_datetime(data)
  if isinstance(data, np.ndarray):
    # Check for outliers
    outliers_mask = np.abs(data - np.mean(data)) > 3 * np.std(data)
    result['outliers'] = bool(np.any(outliers_mask))
    # Check for NaN values
    result['nan values'] = bool(np.isnan(data).any())
  elif isinstance(data, pd.DataFrame):
    # Check for outliers
    outliers_mask = np.abs(data - data.mean()) > 3 * data.std()
    result['outliers'] = bool(np.any(outliers_mask.values))  # type: ignore
    # Check for NaN values
    result['nan values'] = bool(data.isnull().values.any())
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)
  if result['outliers'] == True:
    result['nan values'] = True
  return result


def check_datetime(data: pd.DataFrame | npt.NDArray) -> bool:
  """
  Check if a datetime index or column is present in the given data.

  Parameters
  ----------
  data : pd.DataFrame | np.ndarray
      Input data to check for datetime information.

  Returns
  -------
  bool
      True if datetime information is present, False otherwise.

  Raises
  ------
  ValueError
      Raised when an unsupported data type is provided.
  """

  if isinstance(data, pd.DataFrame):
    return _check_dataframe(data)
  elif isinstance(data, np.ndarray):
    return _check_ndarray(data)
  else:
    raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)


def _check_dataframe(data: pd.DataFrame) -> bool:
  """
  Check if a datetime index or column is present in the pandas DataFrame.

  Parameters
  ----------
  data : pd.DataFrame
      Input DataFrame to check for datetime information.

  Returns
  -------
  bool
      True if datetime information is present, False otherwise.
  """

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
  """
  Check if a datetime column is present in the NumPy array.

  Parameters
  ----------
  data : np.ndarray
      Input NumPy array to check for datetime information.

  Returns
  -------
  bool
      True if datetime information is present, False otherwise.
  """
  df = pd.DataFrame(data)

  for column in df.columns:
    column_type = df[column].dtype
    if pd.api.types.is_datetime64_dtype(column_type):
      return True

  return False


class OutlierRemover():
  """
  Removes outliers from array/dataframe.
  """

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Remove outliers from the data.

    Parameters
    ----------
    data : np.ndarray | pd.DataFrame
        Either numpy array or pandas dataframe.

    Returns
    -------
    np.ndarray | pd.DataFrame
        Array or Dataframe, whichever was given as input.
    """

    data_copy = deepcopy(data)
    values = retrieve_data(data_copy)
    outliers = self.find_outliers(values)
    data_copy = data_copy.astype(float)
    data_copy[outliers] = np.nan
    return data_copy

  def find_outliers(self, values: np.ndarray) -> np.ndarray:
    """
    Use interquartile range to find outliers.

    Parameters
    ----------
    values : np.ndarray
        Input array to find outliers.

    Returns
    -------
    np.ndarray
        Boolean array indicating outliers.
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


@dataclass
class FillMissingData():
  """
  Fills missing values in array/dataframe.
  """
  func: str = 'rollingfill'

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Fill missing values in the data.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Either numpy array or pandas dataframe.
    func : str
        'fillna', 'meanfill' or 'rollingfill'.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Array or DataFrame, whichever you gave it in the first place.

    Raises
    ------
    ValueError
        Raised when an unsupported fill method is provided.
    """
    if self.func == 'dropna':
      data = self.dropna(data)
    elif self.func == 'meanfill':
      data = self.fillna_mean(data)
    elif self.func == 'rollingfill':
      data = self.fillna_rolling(data)
    else:
      raise ValueError(viz_schema.ErrorSchema.FILL_ERROR)

    return data

  def dropna(self,
             data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Drop rows with missing values from the data.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with missing values dropped.

    Raises
    ------
    ValueError
        Raised when an unsupported data type is provided.
    """
    data_copy = deepcopy(data)
    if isinstance(data_copy, pd.DataFrame):
      return data_copy.dropna()
    elif isinstance(data_copy, np.ndarray):
      return data_copy[~np.isnan(data_copy).any(axis=1)]
    else:
      raise ValueError(viz_schema.ErrorSchema.DATA_TYPE)

  def fillna_mean(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Fill missing values in the data with column means.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with missing values filled with column means.

    Raises
    ------
    ValueError
        Raised when an unsupported data type is provided.
    """
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
    """
    Fill missing values in the data using rolling mean.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Input data.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with missing values filled using rolling mean.

    Raises
    ------
    ValueError
        Raised when an unsupported data type is provided.
    """
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


@dataclass
class GenerateDatetime():
  """
  Creates a datetime for dataset or without one.
  """
  start_date: datetime = datetime(2022, 1, 1)
  freq: str = '30T'
  periods: int = 48
  tz: str = 'UTC'

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    """
    Add datetime column to the data.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Either numpy array or pandas dataframe.
    start_date : datetime, optional
        When the datetime series will start, by default datetime(2022, 1, 1).
    freq : str, optional
        Frequency of the datetime series, by default "30T".
    periods : int, optional
        Length used to set the length of the array if no data is given, by default 48.
    tz : str, optional
        Timezone for the datetime series, by default 'UTC'.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Input data with datetime column (index if DataFrame, column zero if array).

    Raises
    ------
    ValueError
        Raised when an unsupported data type is provided.
    """
    data_copy = deepcopy(data)
    if data is not None:
      num_steps = len(data)
    else:
      num_steps = self.periods

    datetime_array = pd.date_range(start=self.start_date,
                                   periods=num_steps,
                                   freq=self.freq,
                                   tz=self.tz)
    if isinstance(data, np.ndarray):
      df = pd.DataFrame(index=datetime_array)
      return np.insert(data_copy, 0, df.index, axis=1)
    elif isinstance(data, pd.DataFrame):
      return data_copy.set_index(datetime_array)
