from enum import Enum, auto


class Seasons(Enum):

  WINTER = auto()
  SPRING = auto()
  SUMMER = auto()
  AUTUMN = auto()


class Frequency(Enum):
  HH = "30T"
  HOUR = "H"
  HALF_DAY = "12H"
  DAY = "D"
  WEEK = "W"
  MONTH = 'M'


class Units(Enum):
  KW = auto()
  KWH = auto()
  KVA = auto()
  GBP = auto()
  GBP_PER_KWH = auto()
  GBP_PER_KW = auto()
  KG_CO2 = auto()
  KG_CO2_PER_KWH = auto()
