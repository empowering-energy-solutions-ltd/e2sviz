from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd

from e2sviz.data import functions
from e2sviz.structure import datetime_schema, enums, viz_schema


def create_seasonal_average_week(season: enums.Season,
                                 dataf: pd.DataFrame,
                                 target_col: str | None = None,
                                 func=np.mean) -> pd.DataFrame:
  """
  Create a seasonal average week for the specified season.

  Arguments:
      season enums.Season:
          The season for which to calculate the average week.  
      dataf pd.DataFrame:
          The input DataFrame.  
      target_col str | None:
          The target column for aggregation. If None, the first column of the DataFrame is used, by default None.  
      func callable:
          The aggregation function to use, by default np.mean.

  Returns:
      The seasonal average week.

  """
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


def seasonal_avg_week_plot_data(
    plot_data: pd.DataFrame
) -> tuple[pd.Index, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Prepare data for the seasonal average week plot.

  Arguments:
      plot_data pd.DataFrame:
          The input DataFrame.

  Returns:
      A tuple containing the datetime index and the average, maximum, and minimum seasonal average week data.

  """
  timestep = enums.TimeStep.HALFHOUR
  avg_data = functions.get_avg_week_by_season_df(
      plot_data, viz_schema.ManipulationSchema.ENERGY, timestep)
  avg_data.index = functions.format_avg_week_index(avg_data, timestep)
  max_data = functions.get_avg_week_by_season_df(
      plot_data, viz_schema.ManipulationSchema.ENERGY, timestep, np.max)
  min_data = functions.get_avg_week_by_season_df(
      plot_data, viz_schema.ManipulationSchema.ENERGY, timestep, np.min)
  datetime = avg_data.index
  return datetime, avg_data, max_data, min_data


def get_seasonal_week(data: pd.DataFrame) -> list[pd.DataFrame]:
  """
  Get weekly data for each season.

  Arguments:
      data pd.DataFrame:
          The input DataFrame.

  Returns:
      A list of DataFrames containing the weekly data for each season.

  """
  data_copy = deepcopy(data)
  loop_data = functions.add_time_features(data_copy)
  output = find_week_in_season(loop_data)
  return output


class ResampleManipulator():
  """
  Resampler class.

  Methods:
    data_formatter:
      Resample the data.

  """

  def data_formatter(self, data: npt.NDArray | pd.DataFrame, split_by: str,
                     aggregation: str) -> npt.NDArray | pd.DataFrame:
    """
    Format the data by resampling.

    Arguments:
        data np.ndarray | pd.DataFrame:
            The input data.  
        split_by str:
            The resampling method.  
        aggregation str:
            The aggregation method.  

    Returns:
        The resampled data.

    """
    data_copy = deepcopy(data)
    if isinstance(data_copy, np.ndarray):
      data_copy = pd.DataFrame(data_copy[:, 1:],
                               index=pd.DatetimeIndex(data_copy[:, 0]))
      return data_copy.resample(split_by).agg(aggregation).to_numpy()
    else:
      return data_copy.resample(split_by).agg(aggregation)


class GroupbyManipulator():
  """
  Manipulate data by grouping.

  Methods:
    data_formatter:
      Format the data by grouping.
  """

  def data_formatter(
      self,
      data: npt.NDArray | pd.DataFrame,
      groupby: list[int | str],
      agg: str,
      target: int | str | None = None) -> npt.NDArray | pd.DataFrame:
    """
    Format the data by grouping.

    Arguments:
        data np.ndarray | pd.DataFrame:
            The input data.  
        groupby list[int | str]:
            The columns to group by.  
        agg str:
            The aggregation method.  
        target int | str | None:
            The column to aggregate, by default None.  

    Returns:
        The grouped data.

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
  Manipulate data by applying equations.

  Methods:
    data_formatter:
      Format the data by applying equations.
  """

  def data_formatter(
      self,
      data: npt.NDArray | pd.DataFrame,
      target_col: str | int,
      func: str,
      new_col: str | float = 'New column') -> npt.NDArray | pd.DataFrame:
    """
    Format the data by applying equations.

    Arguments:
        data np.ndarray  pd.DataFrame:
            The input data.  
        target_col str | int:
            The target column to aggregate.  
        func str:
            The aggregation function as a string.  
        new_col str | float:
            The name for the new column, by default 'New column'.  

    Returns:
        The data with a new column of aggregated values.

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
  Manipulate data by combining columns.

  Methods:
    data_formatter:
      Format the data by combining columns.
  """

  def data_formatter(self, data: npt.NDArray | pd.DataFrame, col_1: str | int,
                     col_2: str | int) -> npt.NDArray | pd.DataFrame:
    """
    Format the data by combining columns.

    Arguments:
        data np.ndarray | pd.DataFrame:
            The input data.  
        col_1 str | int:
            The first column to combine.  
        col_2 str | int:
            The second column to combine.  

    Returns:
        The data with a new column containing the combined values.

    """
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


def find_week_in_season(loop_data: pd.DataFrame) -> list[pd.DataFrame]:
  """
  Find the week in each season of the year.

  Arguments:
      loop_data pd.DataFrame:
          The input DataFrame.

  Returns:
      A list of DataFrames containing the weekly data for each season.

  """
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

  return output
