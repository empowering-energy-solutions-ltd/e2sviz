from dataclasses import dataclass, field
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


init_function_callable = Callable[[npt.NDArray | pd.DataFrame], dict[str,
                                                                     bool]]
data_describer_callable = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class RunVisualisationPreparator:
  data: npt.NDArray | pd.DataFrame
  init_func_test: init_function_callable
  dataprep_outliers: DataPreparationProtocol
  dataprep_nanvals: DataPreparationProtocol
  dataprep_timeseries: DataPreparationProtocol
  data_describer: data_describer_callable
  _described_data: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    return self.run_cleaner()

  def run_cleaner(self) -> None:
    """
    Run the data cleaner.

    Returns
    -------
    None

    """
    self.clean_data()
    self._described_data = self.described_data

  @property
  def described_data(self) -> pd.DataFrame:
    return self.data_describer(pd.DataFrame(self._data))

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