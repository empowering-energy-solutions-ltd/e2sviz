from enum import Enum, StrEnum, auto


class Seasons(Enum):

    WINTER = auto()
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()


class TimeStep(Enum):
    HALFHOUR = auto()
    HOUR = auto()


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
    W = 'W', 'Power'
    WH = 'Wh', 'Energy'
    KVA = 'kVA', 'Power'
    GBP = 'GBP', 'Cost'
    M3 = 'm3', 'Volume'
    GBP_PER_E = 'GBP/', 'Cost per unit energy'
    GBP_PER_P = 'GBP/', 'Cost per unit power'
    G_CO2 = 'gCO2', 'Carbon emissions'
    G_CO2_PER_E = 'gCO2/', 'Carbon emissions per unit energy'
    G_CO2_PER_P = 'gCO2/', 'Carbon emissions per unit energy'
    PERC = '%', 'Percentage'
    DEGC = 'degC', 'Temperature'
    NAN = 'NaN', 'NaN'

    @property
    def label(self) -> str:
        """Get the label used for this unit."""
        return self.value[1]

    @property
    def units(self) -> str:
        """Get the units for which this unit is relevant."""
        return self.value[0]


class Prefix(Enum):
    BASE = '', 1
    KILO = 'k', 2
    MEGA = 'M', 3
    GIGA = 'G', 4
    TERA = 'T', 5

    @property
    def label(self) -> str:
        """Get the label used for this unit."""
        return self.value[0]

    @property
    def index_val(self) -> int:
        """Get the index val for this unit."""
        return self.value[1]


class Season(StrEnum):
    WINTER = auto()
    SUMMER = auto()
    SPRING = auto()
    AUTUMN = auto()
