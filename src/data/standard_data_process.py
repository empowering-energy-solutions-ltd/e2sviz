from dataclasses import dataclass
from typing import Callable, Protocol

# import numpy as np
import numpy.typing as npt
import pandas as pd


class DataPreparationProtocol(Protocol):

  def data_cleaner(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    ...


class DataFormattingProtocol(Protocol):

  def data_formatter(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    ...


@dataclass
class RunVisualisationPreparator:
  data: npt.NDArray | pd.DataFrame
  init_func_test: Callable[[npt.NDArray | pd.DataFrame], dict[str, bool]]
  dataprep_outliers: DataPreparationProtocol
  dataprep_nanvals: DataPreparationProtocol
  dataprep_timeseries: DataPreparationProtocol

  # _data: npt.NDArray | pd.DataFrame | None = None

  def __post_init__(self):
    """
    Perform data preparation steps after object initialization.

    Returns
    -------
    None

    """
    self._prep_check = self.prep_check
    self._data = self.data
    return self.clean_data()

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

  def clean_data(self) -> None:
    """
    Clean the data by applying specific data preparation steps.

    Returns
    -------
    None

    """
    self._data = self.prep_outliers()
    self._data = self.prep_nanvals()
    self._data = self.prep_timeseries()

  def prep_outliers(self) -> npt.NDArray | pd.DataFrame:
    """
    Prepare the data by handling outliers.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with outliers handled.

    """
    if self._prep_check['outliers']:
      return self.dataprep_outliers.data_cleaner(self._data)
    else:
      return self._data

  def prep_nanvals(self) -> npt.NDArray | pd.DataFrame:
    """
    Prepare the data by handling NaN values.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with NaN values handled.

    """
    if self._prep_check['nan values']:
      return self.dataprep_nanvals.data_cleaner(self._data)
    else:
      return self._data

  def prep_timeseries(self) -> npt.NDArray | pd.DataFrame:
    """
    Prepare the data by handling time series data.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Data with time series handling.

    """
    if not self._prep_check['timeseries']:
      return self.dataprep_timeseries.data_cleaner(self._data)
    else:
      return self._data