from dataclasses import dataclass
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd
from IPython.display import display

from src.data import enums, viz_schema


class DataPreparationProtocol(Protocol):

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    ...


class DataFormattingProtocol(Protocol):

  def data_formatter(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    ...


init_function_callable = Callable[[pd.DataFrame], dict[str, bool]]

dataf_callable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class DataPrep:
  data: pd.DataFrame
  init_func_test: init_function_callable
  dataprep_functions: list[dataf_callable] | None = None

  def __post_init__(self):
    """
    Perform data preparation steps after object initialization.

    Returns
    -------
    None

    """

    self.described_raw_data = self.described_data(self.data)
    display(self.described_raw_data)
    if self.dataprep_functions is None:
      print(viz_schema.MessageSchema.NO_DATA_PREP)
      print(self.prep_check)
    else:
      self.clean_data()
      self.described_clean_data = self.described_data(self.cleaned_data)

  @property
  def _data(self) -> pd.DataFrame:
    return self.data

  @property
  def prep_check(self) -> dict[str, bool]:
    """
    Perform initial functional tests on the data.

    Returns
    -------
    dict[str, bool]
        Dictionary containing the results of the functional tests.

    """
    return self.init_func_test(self.data)

  def described_data(self, data: pd.DataFrame) -> pd.DataFrame:
    return self.statistics_of_data(data)

  def clean_data(self) -> None:
    """
    Clean the data by applying specific data preparation steps.

    Returns
    -------
    None

    """
    for functions in self.dataprep_functions:
      self.cleaned_data = functions(self._data)

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
        'Length': len(data)
    }
    describe_df = pd.DataFrame(statistics)
    return describe_df.transpose()


@dataclass
class ColumnMetaData:
  column_name: str
  data: pd.Series
  freq: enums.Frequency
  units: enums.Units

  @property
  def get_x_label(self) -> str:
    return f'Datetime (Timestep:{self.freq.name})'

  @property
  def get_y_label(self) -> str:
    return f'{self.column_name} ({self.units.name})'

  @property
  def get_title(self) -> str:
    return f'{self.get_y_label} vs. {self.get_x_label}'

  @property
  def get_ylim(self) -> tuple[float, float]:
    return (self.data.min() - self.data.min() * 0.1,
            self.data.max() + self.data.max() * 0.1)

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
