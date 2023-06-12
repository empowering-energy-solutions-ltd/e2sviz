import pandas as pd
from e2slib.structures import enums, datetime_schema
from e2slib.data import functions
import numpy as np

def create_seasonal_average_week(season:enums.Season, dataf:pd.DataFrame, target_col_index:str|None=None) -> pd.DataFrame: #, func=np.mean()
    timeseries_data = functions.add_time_features(dataf).copy()
    filt = timeseries_data[
        datetime_schema.DateTimeSchema.SEASON] == season.name
    cols = [
        datetime_schema.DateTimeSchema.DAYOFWEEK,
        datetime_schema.DateTimeSchema.HALFHOUR
    ]
    if target_col_index is None:
        seasonal_data = timeseries_data[filt].groupby(cols).agg({
            timeseries_data.columns[0]:
            'mean'
        })
    else:
        seasonal_data = timeseries_data[filt].groupby(cols).agg({
            target_col_index:
            'mean'
        })
    new_index = functions.format_avg_week_index(seasonal_data,
                                                enums.TimeStep.HALFHOUR)
    seasonal_data.index = new_index
    return seasonal_data

