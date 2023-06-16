import pandas as pd
from e2slib.structures import enums, datetime_schema
from e2slib.utillib import functions
import numpy as np
from typing import Protocol
from data_preperation import retrieve_data

# def create_seasonal_average_week(season:enums.Season, dataf:pd.DataFrame, target_col:str|None=None, func=np.mean) -> pd.DataFrame:
#   timeseries_data = functions.add_time_features(dataf).copy()
#   filt = timeseries_data[
#     datetime_schema.DateTimeSchema.SEASON] == season.name
#   cols = [
#     datetime_schema.DateTimeSchema.DAYOFWEEK,
#     datetime_schema.DateTimeSchema.HALFHOUR
#   ]
#   if target_col is None:
#     target = timeseries_data.columns[0]
#     seasonal_data = timeseries_data[filt].groupby(cols).agg({target: func})
#   else:
#     seasonal_data = timeseries_data[filt].groupby(cols).agg({target_col: func})
#   new_index = functions.format_avg_week_index(seasonal_data,
#                                             enums.TimeStep.HALFHOUR)
#   seasonal_data.index = new_index
#   return seasonal_data

class DataFormattingProtocol(Protocol):
  """
  Formats the data in different ways to allow for various visualisations.
  Can be skipped for plotting full datasets.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """

class AnnualData(DataFormattingProtocol):
  """
  Returns year specific data for choosen year.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame, year:int) -> np.ndarray | pd.DataFrame:
    """
    Takes all data and returns just for specified year
    Parameters:
      Data: Either numpy array or pandas dataframe
      year: Integer of year wanted
    Returns:
      Array or Dataframe, whichever you gave it in the first place.
    """
    if isinstance(data, np.ndarray):
      return data[data[:, 0] == year]
    elif isinstance(data, pd.DataFrame):
      return data[data.index.year == year]
    else:
      raise ValueError("Invalid data type. Expected numpy array or pandas dataframe.")

class SeasonalData(DataFormattingProtocol):
  """
  Formats the data in different ways to allow for various visualisations.
  Can be skipped for plotting full datasets.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """
    

class MonthlyData(DataFormattingProtocol):
  """
  Formats the data in different ways to allow for various visualisations
  Can be skipped for plotting full datasets.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """
    
class WeeklyData(DataFormattingProtocol):
  """
  Formats the data in different ways to allow for various visualisations
  Can be skipped for plotting full datasets.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """

class WorkWeekData(DataFormattingProtocol):
  """
  Formats the data in different ways to allow for various visualisations
  Can be skipped for plotting full datasets.
  """
  def data_formatter(self,
                     data: np.ndarray |
                     pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """ 
    Formating function applied to data in either DataFrame 
    or Array format.
    """
    