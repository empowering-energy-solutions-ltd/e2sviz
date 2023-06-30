class VizSchema:
  PLT = 'plt'
  PLOTLY = 'plotly'


class ManipulationSchema:
  NEW_COL = 'new_col'
  ENERGY = 'Site energy [kWh]'
  POWER = 'Site power [kW]'


class MessageSchema:
  DATA_TYPE = 'Unsupported data type. Please provide a NumPy array or DataFrame.'
  FILL_ERROR = 'Invalid fill method. Please choose "dropna", "meanfill" or "rollingfill".'
  NO_DATA_PREP = 'No data preparation functions provided. Data will not be cleaned. The data check is as follows:'


class FrequencySchema:
  HH = "30T"
  HOUR = "H"
  HALF_DAY = "12H"
  DAY = "D"
  WEEK = "W"
  MONTH = "M"


class UnitsSchema:
  KW = 'kW'
  KWH = 'kWh'
  KVA = 'kVA'
  GBP = 'GBP'
  GBP_PER_KWH = 'GBP/kWh'
  GBP_PER_KW = 'GBP/kW'
  KG_CO2 = 'KgCO2'
  KG_CO2_PER_KWH = 'KgCO2/kWh'