import pandas as pd
from e2slib.structures import enums, datetime_schema
from e2slib.utillib import functions
import numpy as np
from typing import Protocol

def create_seasonal_average_week(season:enums.Season, dataf:pd.DataFrame, target_col:str|None=None, func=np.mean) -> pd.DataFrame:
    timeseries_data = functions.add_time_features(dataf).copy()
    filt = timeseries_data[
        datetime_schema.DateTimeSchema.SEASON] == season.name
    cols = [
        datetime_schema.DateTimeSchema.DAYOFWEEK,
        datetime_schema.DateTimeSchema.HALFHOUR
    ]
    if target_col is None:
        target = timeseries_data.columns[0]
        seasonal_data = timeseries_data[filt].groupby(cols).agg({target: func})
    else:
        seasonal_data = timeseries_data[filt].groupby(cols).agg({target_col: func})
    new_index = functions.format_avg_week_index(seasonal_data,
                                                enums.TimeStep.HALFHOUR)
    seasonal_data.index = new_index
    return seasonal_data

class DataImporter(Protocol):
    def import_data(self, data: pd.DataFrame) -> None:
        pass


class PandasDataImporter:
    def import_data(self, data: pd.DataFrame) -> None:
        # Example implementation: Perform data import operations
        print("Importing data:", data.head())