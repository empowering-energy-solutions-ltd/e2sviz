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

  def viz_type_init(self, data: pd.DataFrame, timeseries: bool,
                    multiple_y: bool):
    ...


class StandardPlot():
  """
  Creates single plot of values either single or double y values.
  """

  def viz_type_init(self, data: pd.DataFrame, timeseries: bool,
                    multiple_y: bool):
    plt.figure(figsize=(12, 6))
    xlabel = "X"
    ylabel = data.columns[0]
    if timeseries & isinstance(data, np.ndarray):
      x = pd.to_datetime(data[:, 0])
    elif timeseries & isinstance(data, pd.DataFrame):
      x = data.index
    if timeseries:
      xlabel = "Datetime"
    plt.title(f"{ylabel} v {xlabel}")
    plt.ylabel(ylabel)
    if multiple_y:
      y_label_1 = data.columns[1]
      plt.plot(x, data.iloc[:, 1], color='red', label=y_label_1)
      plt.title(f"{ylabel}/{y_label_1} v {xlabel}")
      plt.ylabel(f"{ylabel}/{y_label_1}")

    plt.plot(x, data.iloc[:, 0], color='blue', label=ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid()
    plt.show()


class SubplotPlot():
  """
  Create subplots for each column in data.
  """

  def viz_type_init(self, data: pd.DataFrame, timeseries: bool,
                    multiple_y: bool):

    num_cols = data.shape[1]
    fig, axes = plt.subplots(num_cols,
                             1,
                             sharex=True,
                             figsize=(10, num_cols * 4))
    fig.suptitle('Subplots')

    if timeseries & isinstance(data, np.ndarray):
      x = pd.to_datetime(data[:, 0])
    elif timeseries & isinstance(data, pd.DataFrame):
      x = data.index

    for i, column in enumerate(data.columns):
      ax = axes[i] if num_cols > 1 else axes
      ax.plot(x, data[column])
      ax.set_ylabel(column)
      ax.grid()

      ax.xaxis.set_tick_params(which='both', labelbottom=True)

    plt.xlabel("Datetime")
    plt.show()


class BarPlot():
  """
  Creates bar plot from data.
  """

  def viz_type_init(self, data: pd.DataFrame, timeseries: bool,
                    multiple_y: bool):
    column_names = data.columns
    sum_values = data.sum()

    plt.bar(column_names, sum_values)
    plt.xlabel("Columns")
    plt.ylabel("Sum Values")
    plt.title("Sum Values of Each Column")
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


# ------------------------------------------------------------------------------


@dataclass
class Visualizer:
  data: np.ndarray | pd.DataFrame
  timeseries: bool = False
  viz_selector: VizSelector = MatPlotLibSelector()
  viz_type: VizType = StandardPlot()
  multiple_y: bool = False
  columns: list[str] | None = None

  def __post_init__(self):

    plt_settings()
    # self.viz_type.viz_type_init()

  def plot_plt(self):
    """
    Plots the data using matplotlib.
    """
    self.arr_to_dataframe()

    self.viz_type.viz_type_init(self.data, self.timeseries, self.multiple_y)

  def arr_to_dataframe(self):
    """
    Converts numpy array to pandas dataframe.
    """
    if isinstance(self.data, np.ndarray):
      self.data = pd.DataFrame(self.data[:, 1:], index=self.data[:, 0])
      if self.columns:
        self.data.columns = self.columns
