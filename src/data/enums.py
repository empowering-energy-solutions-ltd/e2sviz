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
