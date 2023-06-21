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


# ------------------------------------------------------------------------------


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
    plt.figure(figsize=(12, 6))


class StandardPlot():
  """
  Select visualisation type to be created.
  """

  def viz_type_init(self):
    plt.figure(figsize=(10, 6))


# ------------------------------------------------------------------------------


@dataclass
class Visualizer:
  data: np.ndarray | pd.DataFrame
  timeseries: bool = False
  plot_all: bool = False
  viz_selector: VizSelector = MatPlotLibSelector()
  viz_type: VizType = StandardPlot()
  multiple_y: bool = False
  subplots: bool = False

  def plot_plt(self):
    """
    Plots the data using matplotlib.
    """
    if self.subplots:
      self.plt_subplot_plot()
    else:
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
      plt.title(f"{ylabel} v {xlabel}")
      plt.ylabel(ylabel)
      if self.multiple_y:
        y_label_1 = self.data.columns[1]
        plt.plot(x, self.data.iloc[:, 1], color='red', label=y_label_1)
        plt.title(f"{ylabel}/{y_label_1} v {xlabel}")
        plt.ylabel(f"{ylabel}/{y_label_1}")

      plt.plot(x, self.data.iloc[:, 0], color='blue', label=ylabel)
      plt.xlabel(xlabel)
      plt.legend()
      plt.grid()
      plt.show()

  def plt_subplot_plot(self):
    num_cols = self.data.shape[1]
    fig, axes = plt.subplots(num_cols,
                             1,
                             sharex=True,
                             figsize=(10, num_cols * 4))
    fig.suptitle('Subplots')

    if self.timeseries & isinstance(self.data, np.ndarray):
      x = pd.to_datetime(self.data[:, 0])
    elif self.timeseries & isinstance(self.data, pd.DataFrame):
      x = self.data.index

    for i, column in enumerate(self.data.columns):
      ax = axes[i] if num_cols > 1 else axes
      ax.plot(x, self.data[column])
      ax.set_ylabel(column)
      ax.grid()

      ax.xaxis.set_tick_params(which='both', labelbottom=True)

    plt.xlabel("Datetime")
    plt.show()
