import matplotlib.pyplot as plt
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
  # figsize = (12, 3.5)
  markers = ['x', 'o', '^', 's', '*', 'v']