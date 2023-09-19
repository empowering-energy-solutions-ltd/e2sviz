import numpy as np
import pandas as pd

from e2sviz.structure import datetime_schema, enums

TIMESTEP_SCHEMA_DICT: dict[enums.TimeStep, str] = {
    enums.TimeStep.HALFHOUR: datetime_schema.DateTimeSchema.HALFHOUR,
    enums.TimeStep.HOUR: datetime_schema.DateTimeSchema.HOUR,
}


def get_season(month: int) -> str:
  if 3 <= month <= 5:  #April and May
    return enums.Season.SPRING.name
  elif 6 <= month <= 8:  # June, July, August
    return enums.Season.SUMMER.name
  elif 9 <= month <= 11:  # September, October and November
    return enums.Season.AUTUMN.name
  else:
    return enums.Season.WINTER.name


def add_time_features(dataf: pd.DataFrame) -> pd.DataFrame:
  new_dataf = dataf.copy()
  new_dataf_index: pd.DatetimeIndex = new_dataf.index
  new_dataf[datetime_schema.DateTimeSchema.HOUR] = new_dataf_index.hour
  new_dataf[
      datetime_schema.DateTimeSchema.DAYOFWEEK] = new_dataf_index.dayofweek
  new_dataf[
      datetime_schema.DateTimeSchema.DAYOFYEAR] = new_dataf_index.dayofyear
  new_dataf[datetime_schema.DateTimeSchema.MONTH] = new_dataf_index.month
  new_dataf[datetime_schema.DateTimeSchema.YEAR] = new_dataf_index.year
  new_dataf[datetime_schema.DateTimeSchema.WEEKDAYFLAG] = [
      'weekday' if x < 5 else 'weekend' for x in new_dataf_index.dayofweek
  ]
  new_dataf[datetime_schema.DateTimeSchema.
            HALFHOUR] = new_dataf_index.hour * 2 + new_dataf_index.minute // 30
  new_dataf[datetime_schema.DateTimeSchema.HALFHOUR] = new_dataf[
      datetime_schema.DateTimeSchema.HALFHOUR].astype(int)
  new_dataf[datetime_schema.DateTimeSchema.DATE] = new_dataf_index.date
  new_dataf[
      datetime_schema.DateTimeSchema.WEEK] = new_dataf_index.isocalendar().week
  new_dataf[datetime_schema.DateTimeSchema.SEASON] = new_dataf[
      datetime_schema.DateTimeSchema.MONTH].map(get_season)

  season_dict = {
      enums.Season.WINTER.name: 1,
      enums.Season.SPRING.name: 2,
      enums.Season.SUMMER.name: 3,
      enums.Season.AUTUMN.name: 4
  }
  new_dataf[datetime_schema.DateTimeSchema.SEASON_NUM] = new_dataf[
      datetime_schema.DateTimeSchema.SEASON].map(lambda x: season_dict[x])

  return new_dataf


def get_avg_week_by_season_df(
    dataf: pd.DataFrame,
    target_col: str,
    timestep: enums.TimeStep = enums.TimeStep.HALFHOUR,
    func=np.mean) -> pd.DataFrame:
  groupby_cols = [
      datetime_schema.DateTimeSchema.SEASON,
      datetime_schema.DateTimeSchema.DAYOFWEEK, TIMESTEP_SCHEMA_DICT[timestep]
  ]
  gb_dataf = dataf.groupby(groupby_cols).agg({target_col: func}).unstack(0)
  gb_dataf.columns = [c[1] for c in gb_dataf.columns]
  gb_dataf = gb_dataf[[c.name for c in enums.Season]]
  return gb_dataf


def format_avg_week_index(dataf: pd.DataFrame,
                          timestep: enums.TimeStep) -> pd.Index:
  if timestep is enums.TimeStep.HALFHOUR:
    return dataf.index.get_level_values(0) + (
        (1 / 48) * dataf.index.get_level_values(1))
  else:
    return dataf.index.get_level_values(0) + (
        (1 / 24) * dataf.index.get_level_values(1))