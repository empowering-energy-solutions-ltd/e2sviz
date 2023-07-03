import pandas as pd

import src.chpqa_test_files.schema as schema


def read_data(file_path: str) -> pd.DataFrame:
  return pd.read_csv(file_path)


def compile_data(folder_path: str) -> pd.DataFrame:

  appended_data = []

  for filepath in folder_path.glob(
      "*.csv"
  ):  # Loop to run through each invoice and append to the final outputs.
    #filename = filepath.name
    raw_data = read_data(filepath)
    appended_data.append(raw_data)

  all_raw_data = pd.concat(appended_data, ignore_index=True)

  return all_raw_data


def clean_data(dataf: pd.DataFrame) -> pd.DataFrame:

  dataf.dropna(axis=1, inplace=True)

  empty_col = []
  for column in dataf.columns:
    if dataf[column].max() == 0 and dataf[column].mean() == 0:
      empty_col.append(column)

  dataf.drop(empty_col, axis=1, inplace=True)

  return dataf


def transform_raw_dataf(dataf: pd.DataFrame) -> pd.DataFrame:

  sm3_to_MWh = (39.3 / 3.6) / 1000  #MWh/m3
  chpqa = pd.DataFrame()

  chpqa[schema.outputSchema.Datetime] = pd.to_datetime(
      dataf['From Timestamp'], format='%d/%m/%Y %H:%M:%S')
  chpqa[schema.outputSchema.CHP_elec] = dataf[[
      schema.inputSchema.CHP_elec_day, schema.inputSchema.CHP_elec_night
  ]].sum(axis=1) / 1000
  #,schema.inputSchema.CHP_par_elec_day, schema.inputSchema.CHP_par_elec_night
  chpqa[schema.outputSchema.CHP_heat_total] = dataf[
      schema.inputSchema.CHP_heat]
  #chpqa[schema.outputSchema.CHP_dump_heat] = -dataf[schema.inputSchema.CHP_dump_heat]/1000
  chpqa[schema.outputSchema.CHP_heat] = dataf[
      schema.inputSchema.
      CHP_heat]  #-(dataf[schema.inputSchema.CHP_dump_heat]/1000)
  chpqa[schema.outputSchema.CHP_gas] = dataf[
      schema.inputSchema.CHP_gas]  # * sm3_to_MWh
  chpqa[schema.outputSchema.Boiler_1_heat] = dataf[
      schema.inputSchema.Boiler_1_heat]
  chpqa[schema.outputSchema.Boiler_2_heat] = dataf[
      schema.inputSchema.Boiler_2_heat]
  chpqa[schema.outputSchema.Boiler_3_heat] = dataf[
      schema.inputSchema.Boiler_3_heat]
  chpqa[schema.outputSchema.Boiler_1_gas] = dataf[
      schema.inputSchema.Boiler_1_gas]  # * sm3_to_MWh
  chpqa[schema.outputSchema.Boiler_2_gas] = dataf[
      schema.inputSchema.Boiler_2_gas]  # * sm3_to_MWh
  chpqa[schema.outputSchema.Boiler_3_gas] = dataf[
      schema.inputSchema.Boiler_3_gas]  # * sm3_to_MWh
  chpqa[schema.outputSchema.Total_heat] = (chpqa[[
      schema.outputSchema.CHP_heat, schema.outputSchema.Boiler_1_heat,
      schema.outputSchema.Boiler_2_heat, schema.outputSchema.Boiler_3_heat
  ]].sum(axis=1))
  chpqa[schema.outputSchema.Total_gas] = chpqa[[
      schema.outputSchema.CHP_gas, schema.outputSchema.Boiler_1_gas,
      schema.outputSchema.Boiler_2_gas, schema.outputSchema.Boiler_3_gas
  ]].sum(axis=1)

  chpqa.set_index('Datetime', drop=True, inplace=True)
  chpqa.sort_index(ascending=True, inplace=True)

  return chpqa
