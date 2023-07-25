from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

font_size = 12


def custom_plot(x, y, ax=None, **plt_kwargs):
  if ax is None:
    ax = plt.gca()
  ax.plot(x, y, **plt_kwargs)  ## example plot here
  return ax


def custom_plot_from_df(dataf: pd.DataFrame,
                        ax: plt.Axes = None,
                        **plt_kwargs):
  if ax is None:
    ax = plt.gca()
  dataf.plot(ax=ax, **plt_kwargs)  ## example plot here
  ax.margins(0, None)
  ax.grid()
  ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))


@dataclass
class MatPlotLibPlot():
  container: Optional[plt.Axes] = None

  def __post_init__(self):
    self.plt_settings()
    if self.container is None:
      self.container = plt.figure(figsize=(15, 8)).gca()

  def plt_settings(self):
    """
    Set the plot settings for matplotlib
    """

    plt.rcParams['font.family'] = 'Times New Roman'
    params = {
        'axes.labelsize': font_size + 2,
        'axes.titlesize': font_size + 4,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.size': font_size
    }
    plt.style.use('ggplot')
    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams.update(params)
    fig_format = "png"
    dpi = 1000

  def get_column_kwargs(
      self, column: str, dict_kwargs: dict[str, dict[str,
                                                     str]]) -> dict[str, str]:
    """
    Get the keyword arguments for a column
    
    Parameters
    ----------
    column : str
        Column name
    dict_kwargs : dict[str, dict[str, str]]
        Dictionary containing the plot settings
        
    Returns
    -------
    dict[str, str]
        Dictionary containing the keyword arguments
    """
    return dict_kwargs[column]

  def set_kwargs(self, ax: plt.Axes, kwargs: dict[str, str]):
    """
    Set the keyword arguments for a plot
    
    Parameters
    ----------
    kwargs : dict[str, str]
        Dictionary containing the plot settings
    """
    ax.set_title(kwargs['title'])
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    ax.legend(kwargs['legend'])

  def line_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a single line plot.

    Parameters
    ----------
    dataf : pd.DataFrame
        Data to plot.
    dict_kwargs : dict[str, dict[str, str]]
        Dictionary containing the kwargs for each column.
    **plt_kwargs : dict
        Additional matplotlib plot settings.
    """
    for column in plot_columns:
      ax: plt.Axes = self.container
      kwargs = self.get_column_kwargs(column, dict_kwargs)
      ax = custom_plot(dataf.index, dataf[column], ax=ax)
      self.set_kwargs(ax, kwargs)
      self.container = ax

  def stacked_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                   dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a stacked line plot

    Parameters
    ----------
    x : pd.DatetimeIndex
        x-axis values
    y : pd.Series
        y-axis values
    kwargs : dict
        Dictionary of plot settings
    fig : go.Figure, optional
        Plotly figure, by default None
    **fig_kwargs : dict
        Additional plotly figure settings
    """
    ax = self.container
    cum_sum = pd.Series(0, index=dataf.index)
    # ax.fill_between(dataf.index, zeros, label='zeros', alpha=0)
    # Plot the stacked lines
    for column in plot_columns:
      kwargs = self.get_column_kwargs(column, dict_kwargs)
      self.set_kwargs(ax, kwargs)
      ax.plot(dataf.index, dataf[column] + cum_sum, label=column)
      cum_sum += dataf[column]

    ax.set_title('Stacked Plot')
    ax.legend()

    self.container = ax

  def corr_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a correlation matrix
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    corr_matrix = dataf[plot_columns].corr()
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=None)
    self.container = ax

  def bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a bar plot
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    kwargs : dict[str, str]
        Dictionary containing the plot settings
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    ax = plt.figure(figsize=(15, 8)).gca()
    column_sums = dataf[plot_columns].sum()
    column_names = plot_columns
    sums = column_sums.values.tolist()
    ax.bar(column_names, sums)
    kwargs = self.get_column_kwargs(plot_columns[0], dict_kwargs)
    self.set_kwargs(ax, kwargs)
    ax.set_title('Totals')
    ax.set_xlabel('Columns')

    plt.xticks(rotation=45)
    plt.tight_layout()
    ax.legend().set_visible(False)

    self.container = ax

  def dt_bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                  dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a bar plot

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    kwargs : dict[str, str]
        Dictionary containing the plot settings

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    ax = self.container

    total_bars = len(plot_columns)
    bar_width = 0.8 / total_bars  # Adjust this value as needed to change the bar width

    x_values = np.arange(len(dataf.index))

    for i, col in enumerate(plot_columns):
      kwargs = self.get_column_kwargs(col, dict_kwargs)

      # Calculate the offset for each bar based on its position in the sequence of columns
      bar_offset = (i - total_bars // 2) * bar_width

      ax.bar((x_values + bar_offset) + 1,
             dataf[col],
             label=col,
             width=bar_width)
    # for col in plot_columns:
    #   kwargs = self.get_column_kwargs(col, dict_kwargs)
    #   ax.bar(dataf.index, dataf[col], label=col)

    ax.set_title('Bar plot')
    ax.set_xlabel('Months')  # kwargs['x_label']
    ax.set_ylabel(kwargs['y_label'])
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.xticks(x_values)
    plt.tight_layout()

    self.container = ax

  def box_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a box plot
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    kwargs : dict[str, str]
        Dictionary containing the plot settings
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    ax = plt.figure(figsize=(15, 8)).gca()
    ax.boxplot(dataf[plot_columns].values)
    # ax.set_xticklabels(dataf[plot_columns].columns)
    kwargs = self.get_column_kwargs(plot_columns[0], dict_kwargs)
    self.set_kwargs(ax, kwargs)
    ax.set_title('BoxPlot')
    ax.set_xlabel('Columns')

    plt.xticks(rotation=45)
    plt.tight_layout()
    ax.legend().set_visible(False)

    self.container = ax

  def pie_chart(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a pie chart

    Parameters
    ----------
    data : pd.Series
        Data to plot
    kwargs : dict[str, str]
        Dictionary containing the plot settings

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    column_sums = dataf[plot_columns].sum()
    labels = column_sums.index.tolist()
    values = column_sums.values.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    ax.set_title('Pie Chart')

    self.container = ax

  def show(self) -> Any:
    return self.container

  def save(self, save_path: Path):
    self.container.figure.savefig(save_path, dpi=300)


@dataclass
class PlotlyPlot():

  container: Optional[go.Figure] = None

  def __post_init__(self):
    if self.container is None:
      self.container = go.Figure()
    self.plotly_settings(self.container)

  def plotly_settings(self, fig: go.Figure):
    """
    Set the plotly figure settings

    Parameters
    ----------
    fig : go.Figure
        Plotly figure

    Returns
    -------
    go.Figure
        Plotly figure with settings added.
    """

    fig.update_layout(title=dict(font=dict(size=font_size + 4)),
                      xaxis=dict(title=dict(font=dict(size=font_size)),
                                 tickfont=dict(size=font_size)),
                      yaxis=dict(title=dict(font=dict(size=font_size)),
                                 tickfont=dict(size=font_size)),
                      legend=dict(font=dict(size=font_size)))

    self.container = fig

  def get_column_kwargs(
      self, column: str, dict_kwargs: dict[str, dict[str,
                                                     str]]) -> dict[str, str]:
    """
    Get the keyword arguments for a column
    
    Parameters
    ----------
    column : str
        Column name
    dict_kwargs : dict[str, dict[str, str]]
        Dictionary containing the plot settings
        
    Returns
    -------
    dict[str, str]
        Dictionary containing the keyword arguments
    """
    return dict_kwargs[column]

  def set_kwargs(self, fig: go.Figure, kwargs: dict[str, str]):
    """
    Set the keyword arguments for a plot
    
    Parameters
    ----------
    kwargs : dict[str, str]
        Dictionary containing the plot settings
    """
    fig.update_layout(title=kwargs['title'],
                      xaxis_title=kwargs['x_label'],
                      yaxis_title=kwargs['y_label'])

  def line_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a single line plot

    Parameters
    ----------
    x : pd.DatetimeIndex
        x-axis values
    y : pd.Series
        y-axis values
    kwargs : dict
        Dictionary of plot settings
    fig : go.Figure, optional
        Plotly figure, by default None
    **fig_kwargs : dict
        Additional plotly figure settings
    """
    fig = self.container
    for column in plot_columns:
      kwargs = self.get_column_kwargs(column, dict_kwargs)
      y = dataf[column]
      if isinstance(y, pd.Series):
        fig.add_trace(
            go.Scatter(x=dataf.index,
                       y=y,
                       mode='lines',
                       name=str(kwargs['legend'])))
      else:
        for i, column in enumerate(y.columns):
          trace = go.Scatter(x=dataf.index,
                             y=y[column],
                             mode='lines',
                             name=str(kwargs['legend'][i]))
          fig.add_trace(trace)
      self.set_kwargs(fig, kwargs)

    self.container = fig

  def stacked_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                   dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a stacked line plot

    Parameters
    ----------
    x : pd.DatetimeIndex
        x-axis values
    y : pd.Series
        y-axis values
    kwargs : dict
        Dictionary of plot settings
    fig : go.Figure, optional
        Plotly figure, by default None
    **fig_kwargs : dict
        Additional plotly figure settings
    """
    fig = self.container
    cum_sum = None

    # Plot the stacked lines
    for column in plot_columns:
      kwargs = self.get_column_kwargs(column, dict_kwargs)
      self.set_kwargs(fig, kwargs)
      if cum_sum is None:
        fig.add_trace(
            go.Scatter(x=dataf.index,
                       y=dataf[column],
                       mode='lines',
                       name=column))
        cum_sum = dataf[column]
      else:
        y_values = cum_sum + dataf[column]
        fig.add_trace(
            go.Scatter(x=dataf.index, y=y_values, mode='lines', name=column))
        cum_sum += dataf[column]
    fig.update_layout(title='Stacked Plot')
    self.container = fig

  def corr_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a correlation matrix

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    corr_matrix = dataf[plot_columns].corr()
    df_corr = pd.DataFrame(corr_matrix,
                           columns=corr_matrix.columns,
                           index=corr_matrix.columns)

    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=df_corr.columns,
        y=df_corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),  # Add colorbar with title
        text=np.around(df_corr.values, decimals=2
                       ),  # Use correlation values as text annotations
        hovertemplate=
        'Correlation: %{text}',  # Set hover template to display correlation values
    ))

    fig.update_layout(title='Correlation Plot',
                      xaxis_title='Columns',
                      yaxis_title='Columns')

    self.container = fig

  def bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a bar plot using Plotly

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    plot_columns : list[str]
        Columns to plot
    dict_kwargs : dict[str, dict[str, str]]
        Dictionary containing the plot settings

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    column_sums = dataf[plot_columns].sum()
    sums = column_sums.values.tolist()

    fig = go.Figure(data=[go.Bar(x=plot_columns, y=sums)])
    fig.update_layout(showlegend=False)
    kwargs = self.get_column_kwargs(plot_columns[0], dict_kwargs)

    fig.update_layout(
        barmode='group',
        title='Bar plot',
        xaxis_title='Months',  #kwargs['x_label'],
        yaxis_title=kwargs['y_label'],
        xaxis_tickangle=-45,
        # showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    self.container = fig

  def dt_bar_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
                  dict_kwargs: dict[str, dict[str, str]]):
    """
    Plot a datetime bar plot using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    plot_columns : list[str]
        Columns to plot 
    dict_kwargs : dict[str, dict[str, str]]
        Dictionary containing the plot settings
        
    Returns
    -------
    go.Figure
        Plotly figure
    """

    fig = self.container

    for col in plot_columns:
      kwargs = self.get_column_kwargs(col, dict_kwargs)
      fig.add_trace(go.Bar(x=dataf.index, y=dataf[col], name=col))
    # fig.update_layout(showlegend=True)

    fig.update_layout(
        barmode='group',
        title='Bar plot',
        xaxis=dict(dtick='M1'),
        xaxis_title='Months',  #kwargs['x_label'],
        yaxis_title=kwargs['y_label'],
        xaxis_tickangle=-45,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )
    self.container = fig

  def box_plot(self, dataf: pd.DataFrame, plot_columns: list[str],
               dict_kwargs: dict[str, dict[str, str]]):
    """
    Create a box plot.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    kwargs : dict[str, str]
        Plot settings

    Returns
    -------
    go.Figure
        Plotly figure of box plot
    """
    fig = go.Figure()

    for col in plot_columns:
      kwargs: dict[str,
                   str | list[str]] = self.get_column_kwargs(col, dict_kwargs)
      fig.add_trace(go.Box(y=dataf[col], name=str(kwargs['legend'])))

    fig.update_layout(title='Box Plot',
                      xaxis_title='Columns',
                      yaxis_title=kwargs['y_label'],
                      xaxis=dict(tickangle=45))

    self.container = fig

  def pie_chart(self, dataf: pd.DataFrame, plot_columns: list[str],
                dict_kwargs: dict[str, dict[str, str]]):
    """
    Create a pie chart.

    Parameters
    ----------
    data : pd.Series
        Data to plot
    kwargs : dict[str, str]
        Plot settings

    Returns
    -------
    go.Figure
        Plotly figure of pie chart
    """
    column_sums = dataf[plot_columns].sum()
    labels = column_sums.index.tolist()
    values = column_sums.values.tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title='Pie Chart')

    self.container = fig

  def show(self):
    self.container.show()

  def save(self, save_path: Path):
    self.container.write_html(save_path)
