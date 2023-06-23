from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class DataPreparationProtocol(Protocol):

  def data_cleaner(self) -> npt.NDArray | pd.DataFrame:
    ...


class DataFormattingProtocol(Protocol):

  def data_formatter(
      self, data: npt.NDArray | pd.DataFrame) -> npt.NDArray | pd.DataFrame:
    ...


@dataclass
class RunVisualisationPreperator:
  data: npt.NDArray | pd.DataFrame
  init_func_test: Callable[[npt.NDArray | pd.DataFrame], dict[str, bool]]
  dataprep_outliers: DataPreparationProtocol
  dataprep_nanvals: DataPreparationProtocol
  dataprep_timeseries: DataPreparationProtocol
  _data: npt.NDArray | pd.DataFrame | None = None
  _prep_check: dict[str, bool] | None = None

  def __post_init__(self):
    self._prep_check = self.prep_check
    self._data = self.data
    return self.clean_data()

  @property
  def prep_check(self) -> dict[str, bool]:
    return self.init_func_test(self.data)

  def clean_data(self) -> None:
    self._data = self.prep_outliers()
    self._data = self.prep_nanvals()
    self._data = self.prep_timeseries()
    return self._data

  def prep_outliers(self) -> npt.NDArray | pd.DataFrame:
    if self._prep_check['outliers']:
      return self.dataprep_outliers.data_cleaner(self._data)
    else:
      return self._data

  def prep_nanvals(self) -> npt.NDArray | pd.DataFrame:
    if self._prep_check['nan values']:
      return self.dataprep_nanvals.data_cleaner(self._data, func='fillna')
    else:
      return self._data

  def prep_timeseries(self) -> npt.NDArray | pd.DataFrame:
    if not self._prep_check['timeseries']:
      return self.dataprep_timeseries.data_cleaner(self._data)
    else:
      return self._data

  def run(self) -> str:
    return default_processing(self.data, self.dataprep, self.dataformat)

  # def default_processing(data: pd.DataFrame, dataprep: DataPreparationProtocol,
  #                      dataformat: DataFormattingProtocol) -> str:
  #   return data.pipe(dataprep.data_cleaner).pipe(dataformat.data_formatter)