import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

font_size = 12


def plotly_figure(fig: go.Figure) -> go.Figure:

  fig.update_layout(title=dict(font=dict(size=font_size + 4)),
                    xaxis=dict(title=dict(font=dict(size=font_size)),
                               tickfont=dict(size=font_size)),
                    yaxis=dict(title=dict(font=dict(size=font_size)),
                               tickfont=dict(size=font_size)),
                    legend=dict(font=dict(size=font_size)))

  return fig


def plt_settings():
  plt.rcParams['font.family'] = 'Comic Sans MS'  # 'Times New Roman'
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

  def plot_single(self, x: pd.DatetimeIndex, y: pd.Series, title: str,
                  x_label: str, y_label: str):
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()

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