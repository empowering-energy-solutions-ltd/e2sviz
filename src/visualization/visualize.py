from dataclasses import dataclass
from typing import Protocol

# import matplotlib.pyplot as plt
import pandas as pd

from src.data import viz_schema
from src.visualization.plot_styles import plt_settings

# import plotly.express as px
# from e2slib.structures import enums
# from e2slib.utillib import functions
# from e2slib.visualization import viz_functions


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

  def viz_type_init(self, data: pd.DataFrame):
    ...


@dataclass
class Visualizer:
  data: pd.DataFrame
  viz_type: VizType
  viz_selector: VizSelector = MatPlotLibSelector()
  columns: list[str] | None = None

  def __post_init__(self):

    plt_settings()

  def plot_plt(self):
    """
    Plots the data using matplotlib.
    """

    self.viz_type.viz_type_init(self.data)
