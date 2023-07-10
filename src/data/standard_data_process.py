import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Self  # List,

import numpy as np
import pandas as pd
from e2slib.structures import datetime_schema
from e2slib.utillib import functions
from IPython.display import display

from src.data import enums as viz_enums
from src.data import viz_schema

# InitFuncCallable = Callable[[pd.DataFrame], dict[str, bool]]

DatafCallable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class DataPrep:
  """
  Performs data preparation steps.

  Parameters
  ----------
  data : pd.DataFrame
      The input DataFrame.
  dataprep_functions : list[dataf_callable]
      The list of data preparation functions to be applied to the data.

  Methods
  -------
  described_data(data: pd.DataFrame) -> pd.DataFrame
      Returns a DataFrame containing statistics of the input data.
  clean_data() -> None
      Applies the data preparation functions to the data.
  """
  data: pd.DataFrame
  dataprep_functions: Optional[list[DatafCallable]]

  def __post_init__(self):
    """
    Perform data preparation steps after object initialization.

    Returns
    -------
    None

    """
    self.data = self.data.copy()
    self.described_raw_data = self.described_data(self.data)
    print('Prior to cleaning:')
    display(self.described_raw_data)

    if self.dataprep_functions:
      self.clean_data()
      self.described_clean_data = self.described_data(self.data)
      print('Post cleaning:')
      display(self.described_clean_data)
    else:
      print(viz_schema.MessageSchema.NO_DATA_PREP)

  def described_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing statistics of the input data.

    Parameters
    -------
    data : pd.DataFrame
        The input DataFrame.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated statistics.
    """
    return self.statistics_of_data(data)

  def clean_data(self) -> None:
    """
    Clean the data by applying specific data preparation steps.

    Returns
    -------
    None

    """
    if self.dataprep_functions is not None:
      for prep_function in self.dataprep_functions:
        self.data = prep_function(self.data)
    else:
      pass

  def statistics_of_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for the input data.

    Parameters
    -------
    data : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
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
             dataprep_functions: list[dataf_callable] | None = None) -> Self:
    """
    Concatenate two DataPrep objects.

    Parameters
    -------
    secondary_data : DataPrep
        The secondary DataPrep object to be concatenated to the current one.
    Returns
    -------
    DataPrep object
        The concatenated DataPrep objects.
    """
    combined_data = pd.concat([self.data, secondary_data.data], axis=axis)
    return DataPrep(combined_data, dataprep_functions)


@dataclass
class MetaData:
  """
  Class for storing the data being manipulateds metadata.

  Parameters
  ----------
  metadata : dict[str, dict[str, Any]]
      A dictionary of metadata for each column of data.
  
  Attributes
  ----------
  metadata : dict[str, dict[str, Any]]
      The metadata stored.
  """
  metadata: dict[str, dict[str, Any]]

  def units(self, col: str) -> viz_enums.UnitsSchema:
    """
    Get the units schema of the column data.

    Parameters
    ----------
    col : str
        The column to get the units schema of.
    Returns
    -------
    viz_enums.UnitsSchema
        The units schema of the column data.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.UNITS]

  def siunits(self, col: str) -> viz_enums.SIUnits:
    """
    Get the SI units of the column data.

    Parameters
    ----------
    col : str
        The column to get the SI units of.
    Returns
    -------
    viz_enums.SIUnits
        The SI units of the column data.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.SI]

  def freq(self, col: str) -> viz_schema.FrequencySchema:
    """
    Get the frequency schema of the column data.

    Parameters
    ----------
    col : str
        The column to get the frequency schema of.
    Returns
    -------
    viz_schema.FrequencySchema
        The frequency schema of the column data.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.FREQ]

  def dtype(self, col: str) -> viz_enums.DataType:
    """
    Get the data type of the column.

    Parameters
    ----------
    col : str
        The column to get the data type of.
    Returns
    -------
    viz_enums.DataType
        The data type of the column.
    """
    return self.metadata[col][viz_schema.MetaDataSchema.TYPE]

  def get_x_label(self, col) -> str:
    """
    Get the label for the x-axis of the plot.

    Returns
    -------
    str
        The label for the x-axis of the plot.
    """
    return f'Datetime (Timestep: {self.freq(col)})'

  def get_y_label(self, col: str) -> str:
    """
    Get the label for the y-axis of the plot.

    Parameters
    ----------
    col : str
        The column to get the label for.
    Returns
    -------
    str
        The label for the y-axis of the plot.
    """
    return f'{self.units(col).label} ({self.siunits(col).label}{self.units(col).units})'

  def get_title(self, col: str) -> str:
    """
    Get the title of the plot.

    Parameters
    ----------
    col : str
        The column to get the title for.
    Returns
    -------
    str
        The title of the plot.
    """
    return f'{self.metadata[col][viz_schema.MetaDataSchema.NAME]} vs. {self.get_x_label(col)}'


@dataclass
class DataManip:
  """
  Class for manipulating data.

  Parameters
  ----------
  data : pd.DataFrame
      The input DataFrame.
  frequency : viz_schema.FrequencySchema, optional
      The frequency of the data, by default viz_schema.FrequencySchema.MISSING.
  column_meta_data : MetaData, optional
      The metadata of the data, by default MetaData({}).
  
  Attributes
  ----------
  data : pd.DataFrame
      The input DataFrame.
  frequency : viz_schema.FrequencySchema
      The frequency of the data.
  column_meta_data : MetaData
      The metadata of the data.
  """

  data: pd.DataFrame
  frequency: viz_schema.FrequencySchema = viz_schema.FrequencySchema.MISSING
  column_meta_data: MetaData = field(default_factory=(lambda: MetaData({})))

  def __post_init__(self):
    self.data = self.data.copy()
    self.check_freq()
    self.check_meta_data()
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
    if len(self.column_meta_data.metadata) == 0:
      default_metadata: dict[str, dict[str, Any]] = {}
      for col in self.data.columns:
        default_metadata[col] = {
            viz_schema.MetaDataSchema.NAME: col,
            viz_schema.MetaDataSchema.UNITS: viz_enums.UnitsSchema.NAN,
            viz_schema.MetaDataSchema.SI: viz_enums.UnitsSchema.NAN,
            viz_schema.MetaDataSchema.FREQ: self.frequency,
            viz_schema.MetaDataSchema.TYPE: self.data[col].dtype
        }
      self.column_meta_data = MetaData(default_metadata)

  def val_rescaler(self, column: str, multiplier: float) -> None:
    """
    Rescale the values of the column data.

    Parameters
    ----------
    column : str
        The column to rescale.
    multiplier : float
        The multiplier to rescale the column data by.
    """
    self.data[column] = self.data[column] * multiplier

  def unit_rescaler(self, column: str, step: int) -> None:
    """
    Rescale the units of the column data.

    Parameters
    ----------
    column : str
        The column to rescale.
    step : int
        The step to rescale the column data by.
    """
    si_units_list = list(viz_enums.SIUnits)
    for i, unit in enumerate(viz_enums.SIUnits):
      if unit.index_val == self.column_meta_data.metadata[column][
          viz_schema.MetaDataSchema.SI].index_val:
        next_unit = si_units_list[
            (i + step) %
            len(si_units_list)]  # Get the next unit by using modulo

        self.column_meta_data.metadata[column][
            viz_schema.MetaDataSchema.SI] = next_unit
        break

  def check_rescaling(self):
    """
    Check if the column data requires rescaling.
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
    column_mapping = {
        viz_schema.FrequencySchema.HH: datetime_schema.DateTimeSchema.HALFHOUR,
        viz_schema.FrequencySchema.HOUR: datetime_schema.DateTimeSchema.HOUR,
        viz_schema.FrequencySchema.DAY:
        datetime_schema.DateTimeSchema.DAYOFYEAR,
        viz_schema.FrequencySchema.MONTH: datetime_schema.DateTimeSchema.MONTH
    }
    return column_mapping.get(self.frequency)

  @property
  def dict_of_groupbys(self) -> dict[str, list[str]]:
    freq_col = self.column_from_freq
    return {
        viz_schema.GroupingKeySchema.DAY:
        [datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col],
        datetime_schema.DateTimeSchema.WEEK:
        [datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col],
        datetime_schema.DateTimeSchema.MONTH:
        [datetime_schema.DateTimeSchema.MONTH, freq_col],
        viz_schema.GroupingKeySchema.DAY_SEASON: [
            datetime_schema.DateTimeSchema.SEASON,
            datetime_schema.DateTimeSchema.WEEKDAYFLAG, freq_col
        ],
        viz_schema.GroupingKeySchema.WEEK_SEASON: [
            datetime_schema.DateTimeSchema.SEASON,
            datetime_schema.DateTimeSchema.DAYOFWEEK, freq_col
        ]
    }

  def filter(self,
             year: Optional[list[int]] = None,
             month: Optional[list[int]] = None,
             day: Optional[list[int]] = None,
             hour: Optional[list[int]] = None,
             date: Optional[list[datetime.date]] = None,
             inplace: bool = False) -> pd.DataFrame | None:
    """
    Filter the data by given year, month, day or date.

    Parameters
    -------
    year : list[int]
        The years to filter by.
    month : list[int]
        The months to filter by.
    day : list[int]
        The days to filter by.
    date : list[datetime.date]
        The dates to filter by.
    inplace : bool
        Whether to filter the data in place or return a new DataFrame.
    
    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    None
        If inplace is True filtered data is set to self.data.
    """

    filt = pd.Series(True, index=self.data.index)
    index_data: pd.DatetimeIndex = self.data.index
    if year is not None:
      filt &= index_data.year.isin(year)
    if month is not None:
      filt &= index_data.month.isin(month)
    if day is not None:
      filt &= index_data.day.isin(day)
    if hour is not None:
      filt &= index_data.hour.isin(hour)
    if date is not None:
      filt &= index_data.isin(date)

    filtered_data = self.data.loc[filt].copy()

    if inplace:
      self.data = filtered_data
    else:
      return filtered_data

  def groupby(self,
              groupby_type: str = viz_schema.GroupingKeySchema.WEEK_SEASON,
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> pd.DataFrame | pd.Series:
    """
    Group the data by given column/s and aggregate by a given function.

    Parameters
    ----------
    func : str
        Numpy function to be used for aggregation.
    groupby_type : Callable[[pd.DataFrame], pd.Series]
        The key for dict_of_groupbys used to return groupby columns.

    Returns
    -------
    pd.DataFrame | pd.Series
        The grouped and aggregated data.
    """
    col_list = self.data.columns.tolist()
    timefeature_data = functions.add_time_features(self.data)
    cols = self.dict_of_groupbys.get(groupby_type)
    grouped_data = timefeature_data.groupby(cols).agg(
        {col: func
         for col in col_list})
    new_dict_entry = {viz_schema.MetaDataSchema.INDEX: cols}
    self.column_meta_data.metadata[
        viz_schema.MetaDataSchema.INDEX] = new_dict_entry
    if inplace:
      self.data = grouped_data
    return grouped_data

  def resample(self,
               freq: str = 'D',
               func: Callable[[pd.DataFrame], pd.Series] = np.mean,
               inplace: bool = False) -> pd.DataFrame | pd.Series:
    """
    Resample the data by given frequency and aggregate by a given function.

    Parameters
    ----------
    freq : str
        The frequency to be used for resampling.
    func : Callable[[pd.DataFrame], pd.Series]
        Numpy function to be used for aggregation.
    
    Returns
    -------
    pd.DataFrame | pd.Series
        The resampled and aggregated data.
    """
    resampled_data = self.data.resample(freq).agg(func)
    if inplace:
      self.data = resampled_data
    return resampled_data

  def rolling(self,
              window: int = 3,
              func: Callable[[pd.DataFrame], pd.Series] = np.mean,
              inplace: bool = False) -> pd.DataFrame | pd.Series:
    """
    Rolling window function.

    Parameters
    ----------
    window : int
        The window size.
    func : Callable[[pd.DataFrame], pd.Series]
        Numpy function to be used for aggregation.

    Returns
    -------
    pd.DataFrame | pd.Series
        The rolled and aggregated data.
    """
    rolling_data = self.data.rolling(window).agg(func)
    if inplace:
      self.data = rolling_data
    return rolling_data