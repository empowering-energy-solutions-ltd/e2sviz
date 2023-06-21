import matplotlib.pyplot as plt
import plotly.express as px


class VizSchema:
  PLT = 'plt'
  PLOTLY = 'plotly'


class VizType:
  LINE = 'line'
  SCATTER = 'scatter'
  BAR = 'bar'
  HIST = 'hist'
  BOX = 'box'
  VIOLIN = 'violin'
  PIE = 'pie'
  HEATMAP = 'heatmap'
  CONTOUR = 'contour'


class ManipulationSchema:
  NEW_COL = 'new_col'
  ENERGY = 'Site energy [kWh]'
  POWER = 'Site power [kW]'


class ErrorSchema:
  DATA_TYPE = "Unsupported data type. Please provide a NumPy array or DataFrame."
