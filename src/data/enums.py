from enum import Enum, auto


class Seasons(Enum):

  WINTER = auto()
  SPRING = auto()
  SUMMER = auto()
  AUTUMN = auto()


class DataType(Enum):

  FLOAT = float
  INT = int
  STR = str
  BOOL = bool
  LIST = list
  DICT = dict
  TUPLE = tuple
  SET = set
  NONE = None


class UnitsSchema(Enum):
  KW = 'kW', 'Power'
  KWH = 'kWh', 'Energy'
  MW = 'MW', 'Power'
  MWH = 'MWh', 'Energy'
  KVA = 'kVA', 'Power'
  GBP = 'GBP', 'Cost'
  SM3 = 'sm3', 'Volume'
  GBP_PER_KWH = 'GBP/kWh', 'Cost per unit energy'
  GBP_PER_KW = 'GBP/kW', 'Cost per unit power'
  GBP_PER_MWH = 'GBP/MWh', 'Cost per unit energy'
  GBP_PER_MW = 'GBP/MW', 'Cost per unit power'
  KG_CO2 = 'KgCO2', 'Carbon emissions'
  KG_CO2_PER_KWH = 'KgCO2/kWh', 'Carbon emissions per unit energy'
  KG_CO2_PER_MWH = 'KgCO2/MWh', 'Carbon emissions per unit energy'

  @property
  def label(self) -> str:
    """Get the label used for this unit."""
    return self.value[1]

  @property
  def units(self) -> str:
    """Get the units for which this unit is relevant."""
    return self.value[0]