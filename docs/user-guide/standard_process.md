# Standard Processes

### Class: DataPrep

The `DataPrep` class performs data preparation steps on a given DataFrame.

#### Initialization

```python
@dataclass
class DataPrep:
    data: pd.DataFrame
    init_func_test: Callable[[pd.DataFrame], dict[str, bool]]
    dataprep_functions: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None
```

<b> Parameters: </b><br>
> - `data` (`pd.DataFrame`): The input DataFrame. <br>
> - `init_func_test` (`Callable[[pd.DataFrame], dict[str, bool]]`): A function that performs initial functional tests on the data. <br>
> - `dataprep_functions` (`Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]]`): A list of functions that apply specific data preparation steps. Defaults to None.

#### Methods

##### `__post_init__(self)`

Perform data preparation steps after object initialization.

Returns: <br>
> - None

##### `prep_check(self) -> dict[str, bool]`

Perform initial functional tests on the data.

Returns: <br>
> - `dict[str, bool]`: Dictionary containing the results of the functional tests.

##### `described_data(self, data: pd.DataFrame) -> pd.DataFrame`

Calculate statistics for the input data.

Parameters: <br>
> - `data` (`pd.DataFrame`): The input DataFrame.

Returns: <br>
> - `pd.DataFrame`: A DataFrame containing the calculated statistics.

##### `clean_data(self) -> None`

Clean the data by applying specific data preparation steps.

Returns: <br>
> - None


## Class: ColumnSpecificData

The `ColumnSpecificData` class represents specific data for a column in a DataFrame.

### Initialization

```python
@dataclass
class ColumnSpecificData:
    data: pd.Series
    column_data: dict[str, Any]
```

<b> Parameters: </b> <br>
> - `data` (`pd.Series`): The data for the column.<br>
> - `column_data` (`dict[str, Any]`): Metadata associated with the column.

#### Properties

##### `units` (`viz_schema.UnitsSchema`)

The units of the column. Using viz_schemas UnitsSchema.

Returns: <br>
> - `viz_schema.UnitsSchema`

##### `freq` (`viz_schema.FrequencySchema`)

The frequency of the index for the column.

Returns: <br>
> - `viz_schema.FrequencySchema`

##### `column_name` (`viz_enums.DataType`)

The name of the column.

Returns: <br>
> - `viz_enums.DataType`

##### `get_x_label` (`str`)

Get the x-axis label for plotting.

Returns: <br>
> - `str`

##### `get_y_label` (`str`)

Get the y-axis label for plotting.

Returns: <br>
> - `str`

##### `get_title` (`str`)

Get the title for plotting.

Returns: <br>
> - `str`

##### `get_ylim` (`Tuple[float, float]`)

Get the y-axis limits for plotting.

Returns: <br>
> - `Tuple[float, float]`

#### Methods

##### `plot_all(self) -> None`

Plot the column data.

Returns: <br>
> - None

##### `get_plotting_settings(self) -> None`

Set the plotting settings.

Returns: <br>
> - None
```
