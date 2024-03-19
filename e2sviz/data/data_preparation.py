import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from e2sviz.structure import viz_schema


def convert_data_types(
    data: npt.NDArray | pd.DataFrame,
    columns: list[str] | None = None) -> np.ndarray | ValueError:
  """
  Convert data types between NumPy ndarray and pandas DataFrame.

  Arguments:
      data np.ndarray | pd.DataFrame:
          Input data to be converted.  
      columns List[str] | None:
          List of column names for DataFrame conversion. Defaults to None.  

  Returns:
      Converted data in the specified type.

  Raises:
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
    raise ValueError(viz_schema.MessageSchema.DATA_TYPE)


def retrieve_data(data: npt.NDArray | pd.DataFrame) -> np.ndarray:
  """
  Retrieve the underlying NumPy ndarray from the input data.

  Arguments:
      data np.ndarray | pd.DataFrame:
          Input data from which to retrieve the ndarray.

  Returns:
      np.ndarray: NumPy ndarray representing the input data.

  Raises:
      ValueError: Raised when an unsupported data type is provided.
  """
  if isinstance(data, pd.DataFrame):
    return data.values
  elif isinstance(data, np.ndarray):
    return data


OutlierCallable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class OutlierRemover:
  """
  Removes outliers from dataframe.

  Attributes:
    method str:
        Method to use for outlier removal.

  Methods:
    data_cleaner:
        Remove outliers from the data.
    iqr_outliers:
        Use interquartile range to find outliers.
  """
  method: str = 'iqr'

  def data_cleaner(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the data.

    Arguments:
        data pd.DataFrame:
            Either numpy array or pandas dataframe.

    Returns:
        Dataframe of cleaned data.
    """

    data_copy = deepcopy(data)
    data_with_nans = self.iqr_outliers(data_copy)
    return data_with_nans

  def iqr_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Use interquartile range to find outliers.

    Arguments:
        values np.ndarray:
            Input array to find outliers.

    Returns:
        Boolean array indicating outliers.
    """
    values = retrieve_data(data)
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
    data = data.astype(float)
    data[outliers] = np.nan
    return data


@dataclass
class FillMissingData():
  """
  Fills missing values in dataframe.

  Attributes:
    func str:
        Method to use for filling missing values.

  Methods:
    data_cleaner:
        Fill missing values in the data.
    fillna_mean:
        Fill missing values with column means.
    fillna_rolling:
        Fill missing values using rolling mean.
  """
  func: str = 'rollingfill'

  def data_cleaner(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the data.

    Arguments:
        data pd.DataFrame:
            Pandas dataframe.
        func str:
            `meanfill` or `rollingfill`.

    Returns:
        DataFrame, with values filled.

    Raises:
        ValueError: Raised when an unsupported fill method is provided.
    """
    if self.func == 'meanfill':
      data = self.fillna_mean(data)
    elif self.func == 'rollingfill':
      data = self.fillna_rolling(data)
    else:
      raise ValueError(viz_schema.MessageSchema.FILL_ERROR)

    if data.isnull().values.any():
      data = self.fillna_mean(data)

    return data

  def fillna_mean(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the data with column means.

    Arguments:
        data pd.DataFrame:
            Input data.

    Returns:
        Data with missing values filled with column means.

    Raises:
        ValueError: Raised when an unsupported data type is provided.
    """
    data_copy = deepcopy(data)
    return data_copy.fillna(data_copy.mean())

  def fillna_rolling(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the data using rolling mean.

    Arguments:
        data pd.DataFrame:
            Input data.

    Returns:
        Data with missing values filled using rolling mean.

    Raises:
        ValueError: Raised when an unsupported data type is provided.
    """
    data_copy = deepcopy(data)
    # for column in data_copy.columns:
    # Calculate the rolling mean and fill missing values
    data_copy = data_copy.apply(
        lambda col: col.fillna(col.rolling(window=24, min_periods=1).mean()))
    return data_copy


@dataclass
class ConvertColumnDataFormat():
  """
  Converts column wise data to row wise.

  Attributes:
    freq Optional[str]:
        Frequency of the data, by default '30T'.
    datetime_format Optional[str]:
        Datetime format of the data, by default '%d/%m/%Y'.
    date_column Optional[str]:
        Column name of the date column, by default 'Settlement Date'.

  Methods:
    data_cleaner:
        Reorder data from column wise to row wise.
    convert_half_hourly_to_datetime:
        Convert half hourly data to datetime.
    prep_for_formatting:
        Prepare data for formatting.
  """
  freq: str = '30T'
  datetime_format: str = '%Y-%m-%d'
  date_column: str = 'Settlement Date'

  def data_cleaner(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder data from column wise to row wise.

    Arguments:
        data pd.DataFrame:
            Raw data in a dataframe.

    Returns:
        DataFrame, with values reordered.
    """
    data = data.copy().reset_index()
    df_long = self.prep_for_formatting(data)
    df_long['Datetime'] = df_long['Datetime'].apply(
        self.convert_half_hourly_to_datetime)
    df_long.set_index('Datetime', drop=True, inplace=True)
    df_long.sort_index(inplace=True)
    df_long.rename(columns={'Value': 'Site energy [kWh]'}, inplace=True)
    return df_long

  def convert_half_hourly_to_datetime(self, datetime_str) -> datetime:
    """
    Convert half hourly data to datetime.

    Arguments:
        datetime_str str:
            Half hourly data.

    Returns:
        Datetime object.
    """
    dt = datetime.strptime(datetime_str.split()[0], self.datetime_format)
    half_hourly_value = (int(datetime_str.split()[-1]) - 1)
    minutes_to_add = 30 * half_hourly_value
    dt += timedelta(minutes=minutes_to_add)
    return dt

  def prep_for_formatting(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for formatting.

    Arguments:
        data pd.DataFrame:
            Raw data in a dataframe.

    Returns:
        DataFrame, with values reordered.
    """
    data[self.date_column] = data[self.date_column].astype(str)
    df_long = pd.melt(data,
                      id_vars=[self.date_column],
                      var_name='Timestep',
                      value_name='Value')
    df_long['Datetime'] = df_long[self.date_column] + ' ' + df_long[
        'Timestep'].apply(lambda x: int(re.findall(r'\d+', x)[0])).astype(str)
    df_long.drop(columns=[self.date_column, 'Timestep'], inplace=True)
    return df_long


@dataclass
class GenerateDatetime():
  """
  Creates a datetime for dataset or without one.

  Attributes:
    start_date datetime:
        When the datetime series will start, by default `datetime(2022, 1, 1)`.
    freq str:
        Frequency of the datetime series, by default "30T".
    periods int:
        Length used to set the length of the array if no data is given, by default 48.
    tz str:
        Timezone for the datetime series, by default 'UTC'.

  Methods:
    data_cleaner:
        Add datetime column to the data.
  """
  start_date: datetime = datetime(2022, 1, 1)
  freq: str = '30T'
  periods: int = 48
  tz: str = 'UTC'

  def data_cleaner(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Add datetime column to the data.

    Arguments:
        data pd.DataFrame:
            Either numpy array or pandas dataframe.
        start_date Optional[datetime.datetime]:
            When the datetime series will start, by default datetime(2022, 1, 1).
        freq Optional[str]:
            Frequency of the datetime series, by default "30T".
        periods Optional[int]:
            Length used to set the length of the array if no data is given, by default 48.
        tz Optional[str]:
            Timezone for the datetime series, by default 'UTC'.

    Returns:
        Input data with datetime column (index if DataFrame, column zero if array).

    Raises:
        ValueError: Raised when an unsupported data type is provided.
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

    return data_copy.set_index(datetime_array)
