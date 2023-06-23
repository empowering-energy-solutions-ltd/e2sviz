from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class DataPreparationProtocol(Protocol):

  def data_cleaner(self) -> npt.NDArray | pd.DataFrame:
    ...


class DataFormattingProtocol(Protocol):

  def data_formatter(self) -> npt.NDArray | pd.DataFrame:
    ...


def default_processing(data: pd.DataFrame, dataprep: DataPreparationProtocol,
                       dataformat: DataFormattingProtocol) -> str:
  return data.pipe(dataprep.data_cleaner).pipe(dataformat.data_formatter)