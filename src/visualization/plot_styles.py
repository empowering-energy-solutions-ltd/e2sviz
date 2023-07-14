import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

font_size = 12


def custom_plot(x, y, ax=None, **plt_kwargs):
  if ax is None:
    ax = plt.gca()
  ax.plot(x, y, **plt_kwargs)  ## example plot here
  return (ax)


# def multiple_custom_plots(x, y, ax=None, plt_kwargs={}, sct_kwargs={}):
#   if ax is None:
#     ax = plt.gca()
#   ax.plot(x, y, **plt_kwargs)  #example plot1
#   ax.scatter(x, y, **sct_kwargs)  #example plot2
#   return (ax)


class MatPlotLibPlot():

  def __init__(self):
    self.plt_settings()

  def plt_settings(self):
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
                  kwargs,
                  ax=None,
                  **plt_kwargs):
    plt.figure(figsize=(10, 5))
    custom_plot(x, y, ax=None, **plt_kwargs)
    plt.title(kwargs['title'])
    plt.xlabel(kwargs['x_label'])
    plt.ylabel(kwargs['y_label'])
    if len(kwargs['legend']):
      plt.legend(kwargs['legend'])
    plt.grid()

  def corr_plot(self, corr_matrix: pd.DataFrame) -> None:
    print(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


class PlotlyPlot():

  def plotly_settings(self, fig: go.Figure) -> go.Figure:

    fig.update_layout(title=dict(font=dict(size=font_size + 4)),
                      xaxis=dict(title=dict(font=dict(size=font_size)),
                                 tickfont=dict(size=font_size)),
                      yaxis=dict(title=dict(font=dict(size=font_size)),
                                 tickfont=dict(size=font_size)),
                      legend=dict(font=dict(size=font_size)))

    return fig

  def plot_single(self, x: pd.DatetimeIndex, y: pd.Series, kwargs,
                  **fig_kwargs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(title=kwargs['title'],
                      xaxis_title=kwargs['x_label'],
                      yaxis_title=kwargs['y_label'],
                      **fig_kwargs)
    self.plotly_settings(fig)
    # if len(kwargs['legend']):
    #   fig.update_layout(legend_title_text=kwargs['legend'])
    fig.show()

  def corr_plot(self, corr_matrix: pd.DataFrame):
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

    # Display the plot
    fig.show()
