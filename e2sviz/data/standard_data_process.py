import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Self

import numpy as np
import pandas as pd
from IPython.display import display

from e2sviz.data import functions
from e2sviz.structure import datetime_schema
from e2sviz.structure import enums as viz_enums
from e2sviz.structure import viz_schema

DatafCallable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class DataPrep:
  """
  Performs data preparation steps.

  Attributes:
    data pd.DataFrame:
        The input DataFrame.
    dataprep_functions list[dataf_callable]:
        The list of data preparation functions to be applied to the data.

  Methods:
    described_data:
        Returns a DataFrame containing statistics of the input data.
    clean_data:
        Applies the data preparation functions to the data.
    """
  data: pd.DataFrame
  dataprep_functions: Optional[list[DatafCallable]]
  print_outputs: bool = False

  def __post_init__(self):
    self.data = self.data.copy()

    self.described_raw_data = self.described_data(self.data)
    if self.print_outputs:
      print('Prior to cleaning:')
      display(self.described_raw_data)

    if self.dataprep_functions:
      self.clean_data()
      self.described_clean_data = self.described_data(self.data)
      if self.print_outputs:
        print('Post cleaning:')
        display(self.described_clean_data)

  def described_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
      Returns a DataFrame containing statistics of the input data.

      Arguments:
          data pd.DataFrame:
              The input DataFrame.

      Returns:
          A DataFrame containing the calculated statistics.
      """
    return self.statistics_of_data(data)

  def clean_data(self) -> None:
    """
      Clean the data by applying specific data preparation steps.
      """
    if self.dataprep_functions is not None:
      for prep_function in self.dataprep_functions:
        self.data = prep_function(self.data)
    else:
      pass

  def statistics_of_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
      Calculate statistics for the input data.

      Arguments:
          data pd.DataFrame:
              The input DataFrame.

      Returns:
          A DataFrame containing the calculated statistics.

      """
    statistics = {
        'Count': data.count(),
        'NaN Count': data.isna().sum(),
        'Mean': data.mean(),
        'StD': data.std(),
        'Min': data.min(),
        '1%': data.quantile(0.01),
        '25%': data.quantile(0.25),
        '50%': data.median(),
        '75%': data.quantile(0.75),
        '99%': data.quantile(0.99),
        'Max': data.max(),
        'Range': data.max() - data.min(),
        'Sum': data.sum(),
        'Variance': data.var(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis(),
        'Unique': data.nunique(),
        'Mode': data.mode().iloc[0],
        'Freq': data.groupby(data.columns.tolist()).size().max(),
        'Outliers (3x STD)': (np.abs(data - data.mean())
                              > 3 * data.std()).sum(),
        'Length': len(data)
    }
    describe_df = pd.DataFrame(statistics)
    return describe_df.transpose()

  def concat(self,
             secondary_data: Self,
             axis: Literal[0] | Literal[1] = 1,
             dataprep_functions: list[DatafCallable] | None = None) -> Self:
    """
      Concatenate two DataPrep objects.

      Arguments:
          secondary_data DataPrep:
              The secondary DataPrep object to be concatenated to the current one.

      Returns:
          The concatenated DataPrep objects.
      """
    combined_data = pd.concat([self.data, secondary_data.data], axis=axis)
    return DataPrep(combined_data, dataprep_functions)


@dataclass
class MetaData:
  """
Class for storing the metadata for the dataframe.

Attributes:
  metadata dict[str, dict[str, Any]]:
      A dictionary of metadata for each column of data.

Methods:
  units:
      Get the units schema of the column data.
  siunits:
      Get the SI units of the column data.
  freq:
      Get the frequency schema of the column data.
  dtype:
      Get the data type of the column.
  column_from_freq:
      Get the column name from the frequency.
  get_x_label:
      Get the label for the x-axis of the plot.
  get_y_label:
      Get the label for the y-axis of the plot.
  get_title:
      Get the title of the plot.
  get_legend:
      Get the legend of the plot.
"""
  metadata: dict[str, dict[str, Any]]

  def units(self, col: str) -> viz_enums.UnitsSchema:
    """
    Get the units schema of the column data.

    Arguments:
        col str:
            The column to get the units schema of.

    Returns:
        The units schema of the column data.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.UNITS]

  def siunits(self, col: str) -> viz_enums.Prefix:
    """
    Get the SI units of the column data.

    Arguments:
        col str:
            The column to get the SI units of.

    Returns:
        The SI units of the column data.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.PREFIX]

  @property
  def freq(self) -> viz_schema.FrequencySchema:
    """
    Get the frequency schema of the column data.

    Arguments:
        col str:
            The column to get the frequency schema of.

    Returns:
        The frequency schema of the column data.
    """
    return self.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.FREQ]

  def dtype(self, col: str) -> viz_enums.DataType:
    """
    Get the data type of the column.

    Arguments:
        col str:
            The column to get the data type of.

    Returns:
        The data type of the column.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.TYPE]

  @property
  def column_from_freq(self) -> str:
    column_mapping = {
        viz_schema.FrequencySchema.HH: datetime_schema.DateTimeSchema.HALFHOUR,
        viz_schema.FrequencySchema.HOUR: datetime_schema.DateTimeSchema.HOUR,
        viz_schema.FrequencySchema.DAY:
        datetime_schema.DateTimeSchema.DAYOFYEAR,
        viz_schema.FrequencySchema.MONTH: datetime_schema.DateTimeSchema.MONTH
    }
    return column_mapping.get(self.freq)

  @property
  def get_x_label(self) -> str:
    """
    Get the label for the x-axis of the plot.

    Returns:
        The label for the x-axis of the plot.
    """
    return f'{self.metadata[viz_schema.MetaDataSchema.FRAME][viz_schema.MetaDataSchema.INDEX_COLS][0]} (Timestep: {self.freq})'

  def get_y_label(self, col: str) -> str:
    """
    Get the label for the y-axis of the plot.

    Arguments:
        col str:
            The column to get the label for.

    Returns:
        The label for the y-axis of the plot.
    """
    return f'{self.units(col).label} ({self.siunits(col).label}{self.units(col).units})'

  def get_title(self, col: str, category: str | None = None) -> str:
    """
    Get the title of the plot.

    Arguments:
        col str:
            The column to get the title for.

    Returns:
        The title of the plot.
    """
    freq_col = self.column_from_freq
    dict_for_title = {
        'day of year': [datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col],
        'week of year': [datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col],
        'day of season': [
            datetime_schema.DateTimeSchema.SEASON,
            datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col
        ],
        'week of season': [
            datetime_schema.DateTimeSchema.SEASON,
            datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col
        ]
    }

    if len(self.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS]):
      gb_title: str = ''
      for key, val in dict_for_title.items():
        if val == self.metadata[viz_schema.MetaDataSchema.FRAME][
            viz_schema.MetaDataSchema.GROUPED_COLS]:
          gb_title: str = key
      title = f'{self.metadata[viz_schema.MetaDataSchema.FRAME][viz_schema.MetaDataSchema.GB_AGG]} {gb_title} {self.metadata[col][viz_schema.MetaDataSchema.NAME]} vs. {self.get_x_label}'
      if category is not None:
        title = title + f' - {category}'
    else:
      title = f'{self.metadata[col][viz_schema.MetaDataSchema.NAME]} vs. {self.get_x_label}'
    return title

  def get_legend(self, col: str) -> str:
    """
    Get the legend of the plot.

    Arguments:
        col str:
            The column to get the legend for.

    Returns:
        The legend of the plot.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.LEGEND]


@dataclass
class DataManip:
  """
  Class for manipulating data.

  Attributes:
    data pd.DataFrame:
        The input DataFrame.  
    frequency viz_schema.FrequencySchema:
        The frequency of the data, by default viz_schema.FrequencySchema.MISSING.
    metadata Optional[MetaData]
        The metadata of the data, by default MetaData({}).

  Methods:
    check_freq:
        Check the frequency of the data.
    check_meta_data:
        Check for metadata.
    val_rescaler:
        Rescale the values of the column data.
    unit_rescaler:
        Rescale the units of the column data.
    check_rescaling:
        Check if the column data requires rescaling.
    filter:
        Filter the data by given year, month, day or date.
    groupby:
        Group the data by given column/s and aggregate by a given function.
    populate_legend:
        Populate the legend column of the metadata.
    update_metadata:  
        Update the metadata for the grouped data.
    resampled:  
        Resample the data by given frequency and aggregate by a given function.
    rolling:
        Rolling window function.
    inplace_data: 
        Return the DataManip either as its self or as a new variable.
  """

  data: pd.DataFrame
  frequency: viz_schema.FrequencySchema = viz_schema.FrequencySchema.MISSING
  metadata: Optional[MetaData] = None
  rescale: bool = False

  def __post_init__(self):
    self.data = self.data.copy()
    self.check_freq()
    self.check_meta_data()
    if self.rescale:
      self.check_rescaling()

  def check_freq(self):
    """
    Check the frequency of the data if value not provided,
    it will be infered using pd.infer_freq().
    """
    if self.frequency is viz_schema.FrequencySchema.MISSING:
      self.frequency = pd.infer_freq(self.data.index)

  def check_meta_data(self):
    """
    Check for metadata, if not provided,
    default values will be generated/infered from the data available.
    """
    if self.metadata is None:
      default_metadata: dict[str, dict[str, Any]] = {}
      for col in self.data.columns:
        default_metadata[col] = {
            viz_schema.MetaDataSchema.NAME: col,
            viz_schema.MetaDataSchema.UNITS: viz_enums.UnitsSchema.NAN,
            viz_schema.MetaDataSchema.PREFIX: viz_enums.Prefix.BASE,
            viz_schema.MetaDataSchema.TYPE: self.data[col].dtype,
            viz_schema.MetaDataSchema.LEGEND: col
        }
      default_metadata[viz_schema.MetaDataSchema.FRAME] = {
          viz_schema.MetaDataSchema.FREQ: self.frequency,
          viz_schema.MetaDataSchema.GB_AGG: None,
          viz_schema.MetaDataSchema.INDEX_COLS: [],
          viz_schema.MetaDataSchema.GROUPED_COLS: []
      }
      self.metadata = MetaData(default_metadata)

  def val_rescaler(self, column: str, multiplier: float) -> None:
    """
    Rescale the values of the column data.

    Arguments:
        column str:
            The column to rescale.  
        multiplier float:
            The multiplier to rescale the column data by.
    """
    self.data[column] = self.data[column] * multiplier

  def unit_rescaler(self, column: str, step: int) -> None:
    """
    Rescale the units of the column data.

    Arguments:
        column str:
            The column to rescale.  
        step int:
            The step to rescale the column data by.  
    """
    si_units_list = list(viz_enums.Prefix)
    for i, unit in enumerate(viz_enums.Prefix):
      if unit.index_val == self.metadata.metadata[column][
          viz_schema.MetaDataSchema.PREFIX].index_val:
        next_unit = si_units_list[
            (i + step) %
            len(si_units_list)]  # Get the next unit by using modulo

        self.metadata.metadata[column][
            viz_schema.MetaDataSchema.PREFIX] = next_unit
        break

  def check_rescaling(self):
    """
    Checks if the column data requires rescaling.
    """

    for column in self.data.columns:
      mean_value = self.data[column].mean()
      if mean_value > 1000:
        self.val_rescaler(column, 0.001)
        self.unit_rescaler(column, 1)
      elif mean_value < 0.001:
        self.val_rescaler(column, 1000)
        self.unit_rescaler(column, -1)

  @property
  def column_from_freq(self) -> str:
    """ Get the column name from the frequency.
    
    Returns:
        The column name from the frequency."""
    column_mapping = {
        viz_schema.FrequencySchema.HH: datetime_schema.DateTimeSchema.HALFHOUR,
        viz_schema.FrequencySchema.HOUR: datetime_schema.DateTimeSchema.HOUR,
        viz_schema.FrequencySchema.DAY:
        datetime_schema.DateTimeSchema.DAYOFYEAR,
        viz_schema.FrequencySchema.MONTH: datetime_schema.DateTimeSchema.MONTH
    }
    return column_mapping.get(self.frequency)

  @property
  def dict_of_groupbys(self) -> dict[str, dict[str, list[str]]]:
    """ Get the dictionary of groupby columns.

    Returns:
        The dictionary of groupby columns.
    """
    freq_col = self.column_from_freq
    return {
        viz_schema.GroupingKeySchema.DAY: {
            viz_schema.MetaDataSchema.LEGEND:
            [datetime_schema.DateTimeSchema.WEEKDAYFLAG],
            viz_schema.MetaDataSchema.INDEX_COLS: [freq_col],
            viz_schema.MetaDataSchema.GROUPED_COLS:
            [datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col],
        },
        viz_schema.GroupingKeySchema.WEEK: {
            viz_schema.MetaDataSchema.LEGEND: [],
            viz_schema.MetaDataSchema.INDEX_COLS:
            [datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col],
            viz_schema.MetaDataSchema.GROUPED_COLS:
            [datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col]
        },
        viz_schema.GroupingKeySchema.DAY_SEASON: {
            viz_schema.MetaDataSchema.LEGEND: [
                datetime_schema.DateTimeSchema.SEASON,
                datetime_schema.DateTimeSchema.WEEKDAYFLAG
            ],
            viz_schema.MetaDataSchema.INDEX_COLS: [freq_col],
            viz_schema.MetaDataSchema.GROUPED_COLS: [
                datetime_schema.DateTimeSchema.SEASON,
                datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col
            ]
        },
        viz_schema.GroupingKeySchema.WEEK_SEASON: {
            viz_schema.MetaDataSchema.LEGEND:
            [datetime_schema.DateTimeSchema.SEASON],
            viz_schema.MetaDataSchema.INDEX_COLS:
            [datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col],
            viz_schema.MetaDataSchema.GROUPED_COLS: [
                datetime_schema.DateTimeSchema.SEASON,
                datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col
            ],
        }
    }

  def filter(
      self,
      year: Optional[list[int]] = None,
      season: Optional[list[int]] = None,
      month: Optional[list[int]] = None,
      day: Optional[list[int]] = None,
      hour: Optional[list[int]] = None,
      date: Optional[list[datetime.date]] = None,
      columns: Optional[list[str]] = None,
      inplace: bool = False) -> Self:  # Add value limits e.g above 10kWh?
    """
    Filter the data by given year, month, day or date.

    Arguments:
        year list[int]:
            The years to filter by.  
        season list[int]:
            The seasons to filter by (1: Winter, 2: Spring, 3: Summer, 4: Autumn).  
        month list[int]:
            The months to filter by.  
        day list[int]:
            The days to filter by.  
        date list[datetime.date]:
            The dates to filter by.  
        inplace bool:
            Whether to filter the data in place or return a new DataFrame.  

    Returns:
            The filtered DataFrame. If inplace is True filtered data is set to self.data.  
    """

    filt = pd.Series(True, index=self.data.index)
    index_data: pd.DatetimeIndex = self.data.index
    if year is not None:
      filt &= index_data.year.isin(year)
    if season is not None:
      season_months = {
          1: [12, 1, 2],  # Winter
          2: [3, 4, 5],  # Spring
          3: [6, 7, 8],  # Summer
          4: [9, 10, 11],  # Autumn
      }
      months_to_filter = []
      for s in season:
        months_to_filter.extend(season_months.get(s, []))
      if months_to_filter:
        filt &= index_data.month.isin(months_to_filter)
    if month is not None:
      filt &= index_data.month.isin(month)
    if day is not None:
      filt &= index_data.day.isin(day)
    if hour is not None:
      filt &= index_data.hour.isin(hour)
    if date is not None:
      filt &= index_data.isin(date)

    filtered_data = self.data.loc[filt].copy()
    if columns is not None:
      filtered_data = filtered_data[columns]

    return self.inplace_data(filtered_data, inplace=inplace)

  def groupby(self,
              groupby_type: str = viz_schema.GroupingKeySchema.WEEK_SEASON,
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> Self:
    """
    Group the data by given column/s and aggregate by a given function.

    Arguments:
        func str:
            Numpy function to be used for aggregation.  
        groupby_type Callable[[pd.DataFrame], pd.Series]:
            The key for dict_of_groupbys used to return groupby columns.  

    Returns:
        The grouped and aggregated data.
    """
    col_list = self.data.columns.tolist()
    timefeature_data = functions.add_time_features(self.data)
    gb_col_data = self.dict_of_groupbys[groupby_type]  #.get(groupby_type)
    grouped_data = timefeature_data.groupby(
        gb_col_data[viz_schema.MetaDataSchema.GROUPED_COLS]).agg(
            {col: func
             for col in col_list})
    new_meta_data = self.update_metadata(grouped_data, gb_col_data)
    new_meta_data[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GB_AGG] = func.__name__
    class_meta_data = MetaData(new_meta_data)

    return self.inplace_data(grouped_data,
                             new_meta=class_meta_data,
                             inplace=inplace)

  def populate_legend(self, dataf: pd.DataFrame,
                      gb_col_data: dict[str, list[str]]) -> list[str]:
    """
    Populate the legend column of the metadata.

    Arguments:
        dataf pd.DataFrame:
            The grouped and aggregated data.  
        gb_col_data dict[str, list[str]]:
            The dictionary of groupby columns.  
      
    Returns:
        The list of legend values.
    """

    num_levels = len(gb_col_data[viz_schema.MetaDataSchema.LEGEND])
    unique_values: dict = {}
    for level in range(num_levels):

      level_values = dataf.index.get_level_values(level).unique().to_list()
      unique_values[level] = level_values
    if num_levels == 1:
      result_list = unique_values[0]
    else:
      result_list = [
          f"{value1} {value2}" for value1 in unique_values[0]
          for value2 in unique_values[1]
      ]
    return result_list

  def update_metadata(
      self, grouped_data: pd.DataFrame,
      gb_col_data: dict[str,
                        list[str]]) -> dict[str, dict[str, str | list[str]]]:
    """
    Update the metadata for the grouped data.

    Arguments:
        grouped_data pd.DataFrame:
            The grouped and aggregated data.  
        gb_col_data dict[str, list[str]]:
            The dictionary of groupby columns.  

    Returns:
        The updated metadata.  
    """
    new_meta_data = self.metadata.metadata.copy()
    new_meta_data[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.INDEX_COLS] = gb_col_data[
            viz_schema.MetaDataSchema.INDEX_COLS]
    new_meta_data[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.GROUPED_COLS] = gb_col_data[
            viz_schema.MetaDataSchema.GROUPED_COLS]
    if len(gb_col_data[viz_schema.MetaDataSchema.LEGEND]):
      result_list = self.populate_legend(grouped_data, gb_col_data)
    for c in self.data.columns:
      if len(gb_col_data[viz_schema.MetaDataSchema.LEGEND]):
        new_meta_data[c][viz_schema.MetaDataSchema.LEGEND] = result_list
    return new_meta_data

  def resampled(self,
                freq: str = 'D',
                func: Callable[[pd.DataFrame], pd.DataFrame] = np.mean,
                inplace: bool = False) -> Self:
    """
    Resample the data by given frequency and aggregate by a given function.

    Arguments:
        freq str:
            The frequency to be used for resampling.  
        func Callable[[pd.DataFrame], pd.Series]:
            Numpy function to be used for aggregation.  

    Returns:
        The resampled and aggregated data.
    """
    resampled_data: pd.DataFrame = self.data.resample(freq).agg(func.__name__)
    new_meta_data = deepcopy(self.metadata)
    new_meta_data.metadata[viz_schema.MetaDataSchema.FRAME][
        viz_schema.MetaDataSchema.FREQ] = freq
    return self.inplace_data(resampled_data,
                             freq,
                             new_meta_data,
                             inplace=inplace)

  def rolling(self,
              window: int = 3,
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> Self:
    """
    Rolling window function.

    Arguments:
        window int:
            The window size. 
        func Callable[[pd.DataFrame], pd.Series]:
            Numpy function to be used for aggregation.  

    Returns:
        The rolled and aggregated data.
    """
    rolling_data = self.data.rolling(window).agg(func)
    return self.inplace_data(rolling_data, inplace=inplace)

  def inplace_data(self,
                   new_data: pd.DataFrame,
                   new_freq: Optional[str] = None,
                   new_meta: Optional[MetaData] = None,
                   inplace: bool = False) -> Self:
    """
      Return the DataManip either as its self or as a new variable.

      Arguments:
        data pd.DataFrame:
          The data to be set.
      
      Returns:
          The data.
      """
    if inplace:
      self.data = new_data
      if new_freq:
        self.frequency = new_freq
      if new_meta:
        self.metadata = new_meta
      return Self
    else:
      if new_freq:
        freq = new_freq
      else:
        freq = self.frequency
      if new_meta:
        meta = new_meta
      else:
        meta = self.metadata
      return DataManip(new_data, frequency=freq, metadata=meta)
