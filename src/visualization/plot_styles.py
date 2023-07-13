import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

font_size = 12


def custom_plot(x, y, ax=None, **plt_kwargs):
  if ax is None:
    ax = plt.gca()
  ax.plot(x, y, **plt_kwargs)  ## example plot here
  return (ax)


def multiple_custom_plots(x, y, ax=None, plt_kwargs={}, sct_kwargs={}):
  if ax is None:
    ax = plt.gca()
  ax.plot(x, y, **plt_kwargs)  #example plot1
  ax.scatter(x, y, **sct_kwargs)  #example plot2
  return (ax)


# def plotly_figure(fig: go.Figure) -> go.Figure:

#   fig.update_layout(title=dict(font=dict(size=font_size + 4)),
#                     xaxis=dict(title=dict(font=dict(size=font_size)),
#                                tickfont=dict(size=font_size)),
#                     yaxis=dict(title=dict(font=dict(size=font_size)),
#                                tickfont=dict(size=font_size)),
#                     legend=dict(font=dict(size=font_size)))

#   return fig


def plt_settings():
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


class MatPlotLibPlot():

  def __init__(self):
    plt_settings()

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


class PlotlyPlot():

  def plot_single(self, x: pd.DatetimeIndex, y: pd.Series, kwargs,
                  **fig_kwargs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(title=kwargs['title'],
                      xaxis_title=kwargs['x_label'],
                      yaxis_title=kwargs['y_label'],
                      **fig_kwargs)
    print(kwargs['legend'])
    # if len(kwargs['legend']):
    #   fig.update_layout(legend_title_text=kwargs['legend'])
    fig.show()

  # def plot_multiple(self, x, y, title, x_label, y_label, legend, save_path):
  #   for i in range(len(x)):
  #     plt.plot(x[i], y[i], marker='o')
  #   plt.title(title)
  #   plt.xlabel(x_label)
  #   plt.ylabel(y_label)
  #   plt.legend(legend)
  #   plt.savefig(save_path)
  #   plt.close()

  # def plot_multiple_with_error(self, x, y, y_error, title, x_label, y_label, legend, save_path):
  #   for i in range(le