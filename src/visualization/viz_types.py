import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from e2slib.structures import enums
from e2slib.utillib import functions
from e2slib.visualization import viz_functions

from src.data import viz_schema


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
      plt.plot(x, data.iloc[:, 1], color='mediumorchid', label=y_label_1)
      plt.title(f"{ylabel}/{y_label_1} v {xlabel}")
      plt.ylabel(f"{ylabel}/{y_label_1}")

    plt.plot(x, data.iloc[:, 0], color='royalblue', label=ylabel)
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
      ax.plot(x, data[column], color='royalblue')
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

    plt.bar(column_names, sum_values, color='royalblue')
    plt.xlabel("Columns")
    plt.ylabel("Sum Values")
    plt.title("Sum Values of Each Column")
    plt.xticks(rotation=90)
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
      ax = viz_functions.custom_plot(x, avg_y, ax=ax, color='mediumorchid')
      ax.fill_between(x, y1=min_y, y2=max_y, alpha=0.5, color='royalblue')
      plt.title(f"Average weekly power demand in {temp_season.lower()}")
