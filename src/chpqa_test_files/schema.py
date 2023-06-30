class DataSchema:
  DATE = "Datetime"
  ID = "ID"
  VALUE = "Value"


class MetadataSchema:
  """Schema used to ensure that the metadata information is fully and correclty filled."""
  ENERGY_CARRIER = "energy_carrier"
  UNIT = "Unit"
  ORG_NAME = "original_name"
  ORIGIN_ENTITY_ID = "origin_ID"
  DESTINATION_ENTITY_ID = "destination_ID"  #if origin_ID == destination_ID it is an energy demand
  PROFILE_ID = DataSchema.ID
  TYPE = "Type_of_profile"


class StructureSchema:
  UNIT_ID = "Unit_ID"
  INPUT_ENERGY_CARRIER = "Input_energy_carrier"
  OUTPUT_ENERGY_CARRIER = "Output_energy_carrier"
  ORIGIN_ENTITY_ID = "Origin_entity_ID"
  DESTINATION_ENTITY_ID = "Destination_entity_ID"
  TYPE = "Type"


class ResultsSchema:
  DESTINATION = "Destination"
  ENERGY_CARRIER = "Energy carrier"
  UNIT = "Unit"
  INDEX = "Datetime_UTC"
  NAME = "Name of unit"
  ORIGIN = "Origin"


class TariffSchema:
  YEAR = "Year"
  Month = "Month"
  CCL = "CCL"


class inputSchema:
  Datetime = 'Datetime'
  CHP_elec_day = '{ECHPda} CHP Electricity Generated Day Total (kWh)'
  CHP_elec_night = '{ECHPna} CHP Electricity Generated Night Total (kWh)'
  CHP_par_elec_day = '{PECHPda} CHP Parasitic Electricity Day Total (kWh)'
  CHP_par_elec_night = '{PECHPna} CHP Parasitic Electricity Night Total (kWh)'
  CHP_heat = 'M04 CHP to LTHW Heat Energy Total (MWh)'
  CHP_dump_heat = 'CHP DAC Heat Meter Total (kWh)'
  CHP_gas = 'M15 CHP 2G Natural Gas Corrected Volume Total (Sm3)'
  Boiler_1_heat = 'M01 LTHW Boiler 1 Heat Energy Total (MWh)'
  Boiler_2_heat = 'M02 LTHW Boiler 2 Heat Energy Total (MWh)'
  Boiler_3_heat = 'M03 LTHW Boiler 3 Heat Energy Total (MWh)'
  Boiler_1_gas = 'M10 LTHW Boiler 1 Gas Corrected Volume (Sm3)'
  Boiler_2_gas = 'M11 LTHW Boiler 2 Gas Corrected Volume (Sm3)'
  Boiler_3_gas = 'M12 LTHW Boiler 3 Gas Corrected Volume (Sm3)'


class outputSchema:
  """A schema for the initial test of the data"""
  Datetime = 'Datetime'
  CHP_heat_total = 'CHP_total_heat'
  CHP_elec = 'CHP_electricity'
  CHP_heat = 'CHP_heat'
  CHP_dump_heat = 'CHP_heat_dump'
  CHP_gas = 'CHP_gas'
  Boiler_1_heat = 'Boiler_1_heat'
  Boiler_2_heat = 'Boiler_2_heat'
  Boiler_3_heat = 'Boiler_3_heat'
  Boiler_1_gas = 'Boiler_1_gas'
  Boiler_2_gas = 'Boiler_2_gas'
  Boiler_3_gas = 'Boiler_3_gas'
  Total_heat = 'Total_heat_MWh'
  Total_gas = 'Total_gas_MWh'


class rotherhamUnits:
  CHPPLANT = "CHP_plant"
  BOILER1 = "Boiler_1"
  BOILER2 = "Boiler_2"
  BOILER3 = "Boiler_3"


class CHPQASchema:
  n_power = 'n_power'
  n_heat = 'n_heat'
  qi_val = 'QI'


class xyvalSchema:
  MWe = 'MWe'
  X_coef = 'X_coeff'
  Y_coef = 'Y_coeff'
  key = 'Key_col'


class qualifyingSchema:
  Total_gas = 'Total_gas_MWh'
  Total_heat = 'Total_heat_MWh'
  CHP_elec = 'CHP_electricity'
  n_power = 'n_power'
  n_heat = 'n_heat'
  n_heat_new = 'n_heat_new'
  qi_val = 'QI'
  heat_power_ratio = 'H:P ratio'
  qi_heat = 'qualifying_output_heat'
  qi_power = 'qualifying_output_power'
  qi_fuel = 'qualifying_input_fuel'
