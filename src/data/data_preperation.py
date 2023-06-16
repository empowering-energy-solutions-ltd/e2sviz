from datetime import datetime
import numpy as np
# import numpy.typing as npt
import pandas as pd


class DataPreparationProtocol:
  def data_cleaner(self,
                    data: np.ndarray |
                    pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """
    Cleans the data for manipulation and visualisation.
    """


class OutlierRemover(DataPreparationProtocol):
  """ Removes outliers from array/dataframe """ 
  def data_cleaner(self,
                    data: np.ndarray |
                    pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """
    Remove outliers from the data.
    Parameters:
      Data: Either numpy array or pandas dataframe
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    if isinstance(data, pd.DataFrame):
      values = data.values
    elif isinstance(data, np.ndarray):
      values = data
    else:
      raise ValueError(
        "Unsupported data type. Please provide a NumPy array or DataFrame."
        )
    outliers = self.find_outliers(values)
    data[outliers] = np.nan
    return data

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
      outliers[:, column_idx] = (
        column_data < lower_bound)|(column_data > upper_bound)
    return outliers



class FillMissingData(DataPreparationProtocol):
  """
  Fills missing values in array/dataframe.
  """
  def data_cleaner(self,
                    data: np.ndarray | pd.DataFrame,
                    func: str) -> np.ndarray | pd.DataFrame:
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
      data = self.dropna(data)
    elif func == "fillna":
      data = self.fillna(data)
    else:
      raise ValueError(
        "Invalid fill method. Please choose 'dropna' or 'fillna'."
        )

    return data

  def dropna(self,
             data: np.ndarray |
             pd.DataFrame) -> np.ndarray | pd.DataFrame:
    if isinstance(data, pd.DataFrame):
      return data.dropna()
    elif isinstance(data, np.ndarray):
      return data[~np.isnan(data).any(axis=1)]
    else:
      raise ValueError(
        "Unsupported data type. Please provide a NumPy array or DataFrame."
        )

  def fillna(self,
             data: np.ndarray |
             pd.DataFrame) -> np.ndarray | pd.DataFrame:
    if isinstance(data, pd.DataFrame):
      return data.fillna(data.mean())
    elif isinstance(data, np.ndarray):
      data[np.isnan(data)] = np.nanmean(data)
      return data
    else:
      raise ValueError(
        "Unsupported data type. Please provide a NumPy array or DataFrame."
        )

class GenerateDatetime(DataPreparationProtocol):
  """
  Creates a datetime for dataset or without one.
  """
  def data_cleaner(self,
                    data: np.ndarray | pd.DataFrame | None = None,
                    start_date: datetime = datetime(2022,1,1),
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
        Array or Dataframe, whichever you gave it in the first place.
    """
    if data is None:
      num_steps = periods
    else:
      num_steps = len(data)
    datetime_array = pd.date_range(start=start_date,
                                   periods=num_steps,
                                   freq=freq)
    return datetime_array
