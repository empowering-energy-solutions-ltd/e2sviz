from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
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


class AvgSeasonPlot():
  """
  Creates plot of average values for each season.
  """

  def viz_type_init(self, data: pd.DataFrame, timeseries: bool,
                    multiple_y: bool):
    data = functions.add_time_features(data)
    timestep = enums.TimeStep.HALFHOUR
    avg_data = functions.get_avg_week_by_season_df(
        data, viz_schema.ManipulationSchema.ENERGY, timestep)
    avg_data.index = functions.format_avg_week_index(avg_data, timestep)
    max_data = functions.get_avg_week_by_season_df(
        data, viz_schema.ManipulationSchema.ENERGY, timestep, np.max)
    min_data = functions.get_avg_week_by_season_df(
        data, viz_schema.ManipulationSchema.ENERGY, timestep, np.min)
    x = avg_data.index

    for temp_season in avg_data.columns:
      fig, ax = plt.subplots(figsize=(10, 6))
      avg_y = avg_data[temp_season].values
      max_y = max_data[temp_season].values
      min_y = min_data[temp_season].values
      ax = viz_functions.custom_plot(x, avg_y, ax=ax)
      ax.fill_between(x, y1=min_y, y2=max_y, alpha=0.5)
      plt.title(f"Average weekly power demand in {temp_season.lower()}")
      # set_default_avg_week_labels(ax, timestep, physical_quantity)


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
    self.arr_to_DataFrame()

    self.viz_type.viz_type_init(self.data, self.timeseries, self.multiple_y)

  def arr_to_DataFrame(self):
    """
    Converts numpy array to pandas DataFrame.
    """
    if isinstance(self.data, np.ndarray):
      self.data = pd.DataFrame(self.data[:, 1:], index=self.data[:, 0])
      if self.columns:
        self.data.columns = self.columns
