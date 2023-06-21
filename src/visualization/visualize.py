from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

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


class VizType(Protocol):
  """
  Select visualisation type to be created.
  """

  def viz_type_init(self):
    ...


class AnnualPlot():
  """
  Select visualisation type to be created.
  """

  def viz_type_init(self):
    return plt.figure(figsize=(12, 4))


@dataclass
class Visualizer:
  data: np.ndarray | pd.DataFrame
  timeseries: bool = False
  plot_all: bool = False
  viz_selector: VizSelector = MatPlotLibSelector()
  viz_type: VizType = AnnualPlot()

  def plot_plt(self):
    """
    Plots the data using matplotlib.
    """
    xlabel = "X"
    ylabel = self.data.columns[0]
    if self.timeseries & isinstance(self.data, np.ndarray):
      x = pd.to_datetime(self.data[:, 0])
    elif self.timeseries & isinstance(self.data, pd.DataFrame):
      x = self.data.index
    if self.timeseries:
      xlabel = "Datetime"
    plt_settings()
    self.viz_type.viz_type_init()
    plt.plot(x, self.data.iloc[:, 0], color='blue', label='Data')
    plt.title(f"{ylabel} v {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
