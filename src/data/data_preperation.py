from datetime import datetime
import numpy as np
# import numpy.typing as npt
import pandas as pd

class DataPreparationProtocol:
    def data_modifier(self, data: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """
        Prepare the data for visualization.
        """


class OutlierRemover(DataPreparationProtocol):
    """ Removes outliers from array/dataframe """ 
    def data_modifier(self, data: np.ndarray | pd.DataFrame, 
                     fill_na: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Remove outliers from the data.
        Parameters:
            data (Union[np.ndarray, pd.DataFrame]): The input data to be cleaned.
        Returns:
            Union[np.ndarray, pd.DataFrame]: The cleaned data without outliers.
        """
        if isinstance(data, pd.DataFrame):
            values = data.values
            is_dataframe = True
        elif isinstance(data, np.ndarray):
            values = data
            is_dataframe = False
        else:
            raise ValueError("Unsupported data type. Please provide a NumPy array or DataFrame.")
        outliers = self.find_outliers(values)
        if fill_na:
            if is_dataframe:
                return self.fill_drop_na(data=data, outliers=outliers, func='fillna')
            else:
                values[np.isnan(values)] = np.mean(values)
                return values
        else:
            if is_dataframe:
                return self.fill_drop_na(data=data, outliers=outliers, func='dropna')
            else:
                return values[~np.any(outliers, axis=1)]
            
    def find_outliers(self, values: np.ndarray) -> np.ndarray:
        """
        Find outliers in the data.
        """
        outliers = np.zeros_like(values, dtype=bool)
        for column_idx in range(values.shape[1]):
            column_data = values[:, column_idx]
            q1 = np.percentile(column_data, 25)
            q3 = np.percentile(column_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers[:, column_idx] = (column_data < lower_bound) | (column_data > upper_bound)
        return outliers

    def fill_drop_na(self, data: np.ndarray, outliers: np.ndarray, func: str) -> np.ndarray:
        if func == 'dropna':
            data[outliers] = np.nan
            data = data.dropna()
            return data
        elif func == 'fillna':
            data[outliers] = np.nan
            data = data.fillna(data.mean())
            return data


class GenerateDatetime(DataPreparationProtocol):
    """
    Creates a datetime for dataset or without one.
    """
    def data_modifier(self,
                     data: np.ndarray | pd.DataFrame | None,
                     start_date: datetime,
                     freq: str) -> np.ndarray | pd.DataFrame:
        """
        Create datetime for a dataset or independent of one.
        """

