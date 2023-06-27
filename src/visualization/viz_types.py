from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from e2slib.structures import enums
from e2slib.utillib import functions
from e2slib.visualization import viz_functions

from src.data import manipulations, viz_schema


# --------------- Any time format ---------------
class StandardPlot():
  """
  Creates single plot of values either single or double y values.
  """

  def viz_type_init(self, data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    xlabel = "Datetime"
    ylabel = data.columns[0]
    x = data.index
    plt.title(f"{ylabel} v {xlabel}")
    plt.ylabel(ylabel)
    plt.plot(x, data.iloc[:, 0], color='royalblue', label=ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid()
    plt.show()


class SubplotPlot():
  """
  Create subplots for each column in data.
  """

  def viz_type_init(self, data: pd.DataFrame):
    years = data.index.year.unique()
    for year in years:
      annual_data = data.loc[data.index.year == year]
      num_cols = annual_data.shape[1]
      fig, axes = plt.subplots(num_cols,
                               1,
                               sharex=True,
                               figsize=(10, num_cols * 4))
      fig.suptitle(f'Column data for {year}')
      x = data.index

      for i, column in enumerate(annual_data.columns):
        ax = axes[i] if num_cols > 1 else axes
        ax.plot(x, annual_data[column], color='royalblue')
        ax.set_ylabel(column)
        ax.grid()

        ax.xaxis.set_tick_params(which='both', labelbottom=True)

      plt.xlabel('Datetime')
      plt.show()


@dataclass
class AnnualPlot():
  """
  Creates plot of annual values.
  """
  column: str = 'Site energy [kWh]'

  def viz_type_init(self, data: pd.DataFrame):
    # freq = pd.infer_freq(data.index)
    years = data.index.year.unique()
    for year in years:
      fig, ax = plt.subplots(figsize=(10, 6))
      data_year = data[data.index.year == year]
      data_year[self.column].plot(ax=ax, label='Half-hourly')
      data_year[self.column].resample('d').mean().plot(ax=ax, label='Daily')
      data_year[self.column].resample('w').mean().plot(ax=ax, label='Weekly')
      data_year[self.column].resample('m').mean().plot(ax=ax, label='Monthly')
      ax.set_ylabel(self.column)
      ax.set_xlabel("Date")
      ax.margins(0, None)
      ax.set_title(f'{self.column} for {year}')
      ax.legend(title='Resolution')


# --------------- No time format ---------------


class BarPlot():
  """
  Creates bar plot from data.
  """

  def viz_type_init(self, data: pd.DataFrame):
    column_names = data.columns
    sum_values = data.sum()

    plt.bar(column_names, sum_values, color='royalblue')
    plt.xlabel("Columns")
    plt.ylabel("Sum Values")
    plt.title("Sum Values of Each Column")
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


# --------------- Half-hourly time format ---------------


class ActualSeasonWeekPlot():
  """
  Creates bar plot from data.
  """

  def viz_type_init(self, data: pd.DataFrame):
    weeks = manipulations.get_seasonal_week(data)
    for week in weeks:
      fig, ax = plt.subplots(figsize=(10, 6))
      ax = viz_functions.custom_plot(week.index,
                                     week.iloc[:, 0].values,
                                     ax=ax,
                                     color='royalblue')
      plt.title(f'Given {week["season"].iloc[1]} week power demand')


class AvgSeasonWeekPlot():
  """
  Creates plot of average values for each season.
  """

  def viz_type_init(self, data: pd.DataFrame):

    plot_data = functions.add_time_features(data)
    datetime, avg_data, max_data, min_data = manipulations.seasonal_avg_week_plot_data(
        plot_data)

    for temp_season in avg_data.columns:
      fig, ax = plt.subplots(figsize=(10, 6))
      avg_y = avg_data[temp_season].values
      max_y = max_data[temp_season].values
      min_y = min_data[temp_season].values
      ax = viz_functions.custom_plot(datetime, avg_y, ax=ax, color='cyan')
      ax.fill_between(datetime,
                      y1=min_y,
                      y2=max_y,
                      alpha=0.5,
                      color='royalblue')
      plt.title(f'Average weekly power demand in {temp_season.lower()}')


class AnnualSeasonalWeekPlot():

  def viz_type_init(self, data: pd.DataFrame):
    featured_data = functions.add_time_features(data)
    years = featured_data.index.year.unique()
    for temp_year in years:
      data_year = featured_data[featured_data.index.year == temp_year]
      avg_data = functions.get_avg_week_by_season_df(
          data_year, viz_schema.ManipulationSchema.ENERGY,
          enums.TimeStep.HALFHOUR)
      # avg_data.columns = avg_data.columns.get_level_values(1)
      avg_data.index = functions.format_avg_week_index(avg_data,
                                                       enums.TimeStep.HALFHOUR)
      # avg_data.plot()
      fig, ax = plt.subplots(figsize=(10, 6))
      ax = viz_functions.custom_plot_from_df(avg_data, ax=ax, linewidth=1)

      plt.xlabel('Day of the week (Timestep: 30 minutes)')
      plt.ylabel('Energy profile (kWh)')
      plt.title(f'Average weekly energy profile - {temp_year}')
      plt.ylim(0, 40)
      plt.show()
