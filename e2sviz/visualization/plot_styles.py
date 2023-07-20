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


class MatPlotLibPlot():

  def __init__(self):
    self.plt_settings()

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
    plt.rcParams.update(params)
    fig_format = "png"
    dpi = 1000

  def plot_single(self,
                  x: pd.DatetimeIndex | pd.Series,
                  y: pd.Series,
                  kwargs: dict[str, str],
                  fig_ax: Optional[plt.Axes] = None,
                  **plt_kwargs) -> plt.Figure:
    """
    Plot a single line plot

    Parameters
    ----------
    x : pd.DatetimeIndex
        x-axis values
    y : pd.Series
        y-axis values
    kwargs : dict
        Dictionary containing the plot settings
    
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    if fig_ax is None:
      fig, ax = plt.subplots(figsize=(10, 5))
    else:
      ax = fig_ax
      fig = ax.get_figure()

    ax = custom_plot(x, y, ax=ax, **plt_kwargs)

    ax.set_title(kwargs['title'])
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    if len(kwargs['legend']):
      ax.legend(kwargs['legend'])
    ax.grid()
    return fig

  def corr_plot(self, corr_matrix: pd.DataFrame) -> plt.Figure:
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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    return fig

  def bar_plot(self, data: pd.DataFrame, kwargs: dict[str, str],
               cols: list[str], sum_vals: bool) -> plt.Figure:
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

    fig, ax = plt.subplots(figsize=(10, 5))
    if sum_vals:
      column_sums = data[cols].sum()
      column_names = cols
      sums = column_sums.values.tolist()
      ax.bar(column_names, sums, ax=ax)
    else:
      data[cols].plot(kind='bar', ax=ax)

    ax.set_title(kwargs['title'])
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])
    ax.legend(kwargs['legend'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()

    return fig

  def box_plot(self, data: pd.DataFrame, kwargs: dict[str, str]) -> plt.Figure:
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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data.values)
    ax.set_xticklabels(data.columns)
    ax.set_title(kwargs['title'])
    ax.set_xlabel(kwargs['x_label'])
    ax.set_ylabel(kwargs['y_label'])

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()

    return fig

  def pie_chart(self, data: pd.DataFrame, kwargs: dict[str,
                                                       str]) -> plt.Figure:
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
    column_sums = data.sum()
    labels = column_sums.index.tolist()
    values = column_sums.values.tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    ax.set_title(kwargs['title'])

    return fig


class PlotlyPlot():

  def plotly_settings(self, fig: go.Figure) -> go.Figure:
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

    return fig

  def plot_single(self,
                  x: pd.DatetimeIndex | pd.Series,
                  y: pd.Series,
                  kwargs,
                  fig_ax: Optional[go.Figure] = None,
                  **fig_kwargs) -> go.Figure:
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
    if fig_ax is None:
      fig = go.Figure()
    else:
      fig = fig_ax
    if isinstance(y, pd.Series):
      fig.add_trace(
          go.Scatter(x=x, y=y, mode='lines', name=str(kwargs['legend'])))
    else:
      for i, column in enumerate(y.columns):
        trace = go.Scatter(x=x,
                           y=y[column],
                           mode='lines',
                           name=str(kwargs['legend'][i]))
        fig.add_trace(trace)

    # fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(title=kwargs['title'],
                      xaxis_title=kwargs['x_label'],
                      yaxis_title=kwargs['y_label'],
                      **fig_kwargs)
    self.plotly_settings(fig)

    return fig

  def corr_plot(self, corr_matrix: pd.DataFrame) -> go.Figure:
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

    return fig

  def bar_plot(self, data: pd.DataFrame, kwargs: dict[str, str],
               cols: list[str], sum_vals: bool):
    """
    Plot a bar plot using Plotly

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    kwargs : dict[str, str]
        Dictionary containing the plot settings
    cols : list[str]
        List of column names to plot
    sum_vals : bool
        Whether to plot the sum of values in columns or individual columns

    Returns
    -------
    go.Figure
        Plotly figure
    """

    if sum_vals:
      column_sums = data[cols].sum()
      sums = column_sums.values.tolist()

      fig = go.Figure(data=[go.Bar(x=cols, y=sums)])
      fig.update_layout(showlegend=False)
    else:
      fig = go.Figure()

      for col in cols:
        fig.add_trace(go.Bar(x=data.index, y=data[col], name=col))
      fig.update_layout(showlegend=True)

    fig.update_layout(
        barmode='group',
        title=kwargs['title'],
        xaxis_title=kwargs['x_label'],
        yaxis_title=kwargs['y_label'],
        xaxis_tickangle=-45,
        # showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig

  def box_plot(self, data: pd.DataFrame, kwargs: dict[str, str]) -> go.Figure:
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

    for col in data.columns:
      fig.add_trace(go.Box(y=data[col], name=col))

    fig.update_layout(title=kwargs['title'],
                      xaxis_title=kwargs['x_label'],
                      yaxis_title=kwargs['y_label'],
                      xaxis=dict(tickangle=45))

    return fig

  def pie_chart(self, data: pd.DataFrame, kwargs: dict[str, str]) -> go.Figure:
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
    column_sums = data.sum()
    labels = column_sums.index.tolist()
    values = column_sums.values.tolist()

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title=kwargs['title'])

    return fig
