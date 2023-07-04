from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from e2slib.utillib import functions
from IPython.display import display

from src.data import enums as viz_enums
from src.data import viz_schema

init_function_callable = Callable[[pd.DataFrame], dict[str, bool]]

dataf_callable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class DataPrep:
  data: pd.DataFrame
  dataprep_functions: list[dataf_callable] | None = None

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

    if self.dataprep_functions is not None:
      self.clean_data()
      self.described_clean_data = self.described_data(self.data)
      print('Post cleaning:')
      display(self.described_clean_data)
    else:
      print(viz_schema.MessageSchema.NO_DATA_PREP)

  def described_data(self, data: pd.DataFrame) -> pd.DataFrame:
    return self.statistics_of_data(data)

  def clean_data(self) -> None:
    """
    Clean the data by applying specific data preparation steps.

    Returns
    -------
    None

    """
    if self.dataprep_functions is not None:
      for functions in self.dataprep_functions:
        self.data = functions(self.data)
    else:
      pass

  def statistics_of_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for the input data.

    Parameters
    ----------
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
    ----------
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
class DataManip:
  data: pd.DataFrame
  column_meta_data: dict[str, dict[str, Any]] = field(default_factory=dict)

  def __post_init__(self):
    self.data = self.data.copy()
    self.data = functions.add_time_features(self.data)

  @property
  def dict_of_groupbys(self) -> dict[str, list[str]]:
    return {'day': ['dayofweek', 'dayofyear']}

  def filter(self):
    pass

  def groupby(self):
    pass

  def resample(self):
    pass

  def rolling(self):
    pass


@dataclass
class ColumnVizData:
  data: pd.Series
  column_data: dict[str, Any]

  @property
  def units(self) -> viz_enums.UnitsSchema:
    return self.column_data['Units']

  @property
  def freq(self) -> viz_schema.FrequencySchema:
    return self.column_data['Freq']

  @property
  def column_name(self) -> viz_enums.DataType:
    return self.column_data['Name']

  @property
  def get_x_label(self) -> str:
    return f'Datetime (Timestep: {self.freq})'

  @property
  def get_y_label(self) -> str:
    return f'{self.units.label} ({self.units.units})'

  @property
  def get_title(self) -> str:
    return f'{self.column_name} vs. {self.get_x_label}'

  @property
  def get_ylim(self) -> tuple[float, float]:
    return (self.data.min() - (self.data.max() * 0.1),
            self.data.max() + (self.data.max() * 0.1))

  def plot_all(self) -> None:
    plt.figure(figsize=(15, 5))
    plt.plot(self.data.index, self.data.values)
    self.get_plotting_settings()
    plt.grid()

  def get_plotting_settings(self) -> None:
    plt.xlabel(self.get_x_label)
    plt.ylabel(self.get_y_label)
    plt.title(self.get_title)
    plt.ylim(self.get_ylim)


def generate_column_classes(df, column_metadata):
  column_classes = []

  for i, column in enumerate(df.columns):
    # class_name = column.capitalize().replace(' ', '')
    column_key = f'column_{i + 1}'
    column_key_data = column_metadata[column_key]

    # Define the class dynamically
    cls = ColumnVizData(df[column], column_key_data)

    column_classes.append(cls)

  return column_classes
