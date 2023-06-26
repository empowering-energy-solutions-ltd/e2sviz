from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from e2slib.structures import enums
from e2slib.utillib import functions
from e2slib.visualization import viz_functions

from src.data import viz_schema
from src.visualization.plot_styles import plt_settings


class VizSelector(Protocol):
  """
  Selects the visualisation library to be used.
  """

  def visualizer_init(self) -> str:
    ...


class MatPlotLibSelector():
  """
  Selects the visualisation library to matplotlib.
  """

  def visualizer_init(self) -> str:
    return viz_schema.VizSchema.PLT


class PlotlySelector():
  """
  Selects the visualisation library to plotly.
  """

  def visualizer_init(self) -> str:
    return viz_schema.VizSchema.PLOTLY


# ------------------------------------------------------------------------------


class VizType(Protocol):
  """
  Select visualisation type to be created.
  """

  def viz_type_init(self, data: pd.DataFrame):
    ...


@dataclass
class Visualizer:
  data: npt.NDArray | pd.DataFrame
  viz_type: VizType
  timeseries: bool = False
  viz_selector: VizSelector = MatPlotLibSelector()
  multiple_y: bool = False
  columns: list[str] | None = None

  def __post_init__(self):

    plt_settings()
    # self.viz_type.viz_type_init()

  def plot_plt(self):
    """
    Plots the data using matplotlib.
    """
    self.arr_to_DataFrame()

    self.viz_type.viz_type_init(self.data)

  def arr_to_DataFrame(self):
    """
    Converts numpy array to pandas DataFrame.
    """
    if isinstance(self.data, np.ndarray):
      self.data = pd.DataFrame(self.data[:, 1:], index=self.data[:, 0])
      if self.columns:
        self.data.columns = self.columns
