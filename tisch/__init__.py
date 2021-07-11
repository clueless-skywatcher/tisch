import numpy as np
from IPython.core.display import HTML, display

__version__ = '0.0.1'

class DataTable:

    def __init__(self, data):
        """
        A DataTable denotes a table of values, and the values can be of any type.
        DataTable is created by passing a dictionary of keys and a list of values
        for those keys.

        Parameters:
        -----------
        data: dict
            A dictionary of string keys, each mapped to a NumPy Array 
        """

        self._check_input_type(data)
        self._check_array_length(data)

        self._data = self._convert_unicode_to_object(data)


    def _check_input_type(self, data):
        if not isinstance(data, dict):
            raise TypeError("Input type is not a dictionary")
        
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"Dictionary keys must be a string")            
            if not isinstance(value, np.ndarray):
                raise TypeError(f"Dictionary values must be a NumPy array")
            if value.ndim != 1:
                raise ValueError(f"Dictionary values must be a 1-dimensional array")

    def _check_array_length(self, data):
        for i, value in enumerate(data.values()):
            if i == 0:
                arr_len = len(value)
            elif arr_len != len(value):
                raise ValueError("All arrays must be of the same length") 

    def _convert_unicode_to_object(self, data):
        updated_data = {}
        for k, v in data.items():
            if v.dtype.kind == 'U':
                updated_data[k] = v.astype('object')
            else:
                updated_data[k] = v
        return updated_data

    def __len__(self):
        return len(next(iter(self._data.values())))

    @property
    def columns(self):
        """
        Gives a list of the column names for the current DataTable.

        Returns
        -------
        A list of column names
        """
        return list(self._data)

    @columns.setter
    def columns(self, cols):
        """
        Takes in a list of columns, which is of the same length of the
        current number of columns in the DataTable, and replaces the old
        column names with new column names.

        Parameters
        ----------
        cols: A list of strings

        Returns
        -------
        None
        """
        if not isinstance(cols, list):
            raise TypeError("Columns must be a list")
        
        if len(cols) != len(self._data):
            raise ValueError("New column list must be of same length as old list")
        
        for s in cols:
            if not isinstance(s, str):
                raise TypeError("Column names must be strings")

        if len(cols) != len(set(cols)):
            raise ValueError("Column names must not have duplicates")

        self._data = dict(zip(cols, self._data.values()))

    @property
    def shape(self):
        """
        Returns
        -------
        A tuple consisting of number of rows and columns
        """
        return len(self), len(self._data)

    def _repr_html_(self):
        html = '<table><thead><tr><th></th>'
        for col in self.columns:
            html += f"<th>{col:10}</th>"
        
        html += '</tr></thead>'
        html += '<tbody>'

        only_head = False
        num_head = 10
        num_tail = 10

        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f'<tr><td><strong>{i}</strong></td>'
            for col, values in self._data.items():
                kind = values.dtype.kind

                if kind == 'f':
                    html += f'<td>{values[i]:10.3f}</td>'
                elif kind == 'b':
                    html += f'<td>{values[i]}</td>'
                elif kind == 'O':
                    v = values[i]
                    if v is None:
                        v = 'None'
                    html += f'<td>{v:10}</td>'
                else:
                    html += f'<td>{values[i]:10}</td>'
            html += f'</tr>'

        if not only_head:
            html += f'<tr><strong><td>...</td></strong>'
            for i in range(len(self.columns)):
                html += '<td>...</td>'
            html += f'</tr>'

            for i in range(-num_tail, 0):
                html += f'<tr><td><strong>{len(self) + i}</strong></td>'
                for col, values in self._data.items():
                    kind = values.dtype.kind

                    if kind == 'f':
                        html += f'<td>{values[i]:10.3f}</td>'
                    elif kind == 'b':
                        html += f'<td>{values[i]}</td>'
                    elif kind == 'O':
                        v = values[i]
                        if v is None:
                            v = 'None'
                        html += f'<td>{v:10}</td>'
                    else:
                        html += f'<td>{values[i]:10}</td>'
                html += f'</tr>'
        
        html += '</tbody></table>'
        return html

    @property
    def values(self):
        """
        Returns
        -------
        A 2D numpy array of values in the DataTable
        """
        return np.column_stack(list(self._data.values()))

    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataTable of column names in one columns and
        their data types in the other
        """
        DTYPE_NAMES = {
            "O": "string",
            "i": "integer",
            "f": "float",
            "b": "boolean"
        }

        colnames = np.array(list(self._data.keys()))
        
        dtypes_ = np.array([DTYPE_NAMES[v.dtype.kind] for v in self._data.values()])

        return DataTable({
            'Column Name': colnames,
            'Data Type': dtypes_
        })

    def __getitem__(self, index):
        """
        Use the brackets operator to select rows and columns
        Usage:
        ------
        df['col1'] ---> Selects only 'col1'
        df[['col1', 'col2']] ---> Selects both 'col1' and 'col2'
        df[boolean] ---> Selects a DataTable that contains a boolean condition
        df[rs, cs] ---> Row and column selected together

        Returns:
        --------
        A subset of the original DataTable based on the indices provided
        """

        if isinstance(index, str):
            return DataTable({
                index: self._data[index]
            })
        
        if isinstance(index, list):
            return DataTable({
                col: self._data[col] for col in index
            })

        if isinstance(index, DataTable):
            if index.shape[1] != 1:
                raise ValueError('Index must be a one-column DataTable')
            
            a = next(iter(index._data.values()))

            if a.dtype.kind != 'b':
                raise ValueError('Item must be a one-column Boolean DataTable')

            return DataTable({
                col: value[a] for col, value in self._data.items()
            })

        if isinstance(index, tuple):
            return self._getitem_tuple(index)

        raise TypeError("Wrong data type entered. Pass either a string, tuple, list or DataTable")

    def _getitem_tuple(self, index):
        if len(index) != 2:
            raise TypeError("Tuple must have length of exactly 2")
        
        row, col = index

        if isinstance(row, int):
            row = [row]
        elif isinstance(row, DataTable):
            if row.shape[1] != 1:
                raise ValueError("Row selection must be of 1 column")
            row = next(iter(row._data.values()))
            if row.dtype.kind != 'b':
                raise TypeError('Row selection must be a boolean DataTable')
        elif not isinstance(row, (list, slice)):
            raise TypeError("Row selection is not a list, slice, int or DataTable")

        
        if isinstance(col, int):
            col = [self.columns[col]]        
        elif isinstance(col, str):
            col = [col]
        elif isinstance(col, list):
            new_cols = []
            for c in col:
                if isinstance(c, int):
                    new_cols.append(self.columns[c])
                elif isinstance(c, str):
                    # Assume c is a string
                    new_cols.append(c)
                
            col = new_cols
        elif isinstance(col, slice):
            start = col.start
            stop = col.stop
            step = col.step

            if isinstance(start, str):
                start = self.columns.index(start)

            if isinstance(stop, str):
                stop = self.columns.index(stop) + 1
            
            col = self.columns[start:stop:step]

        else:
            raise TypeError("Column selection must be slice, int, list or str")

        data = {}
        for c in col:
            data[c] = self._data[c][row]

        return DataTable(data)

    def _ipython_key_completions_(self):
        return self.columns

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError("Value must be one-dimensional array")
            if len(value) != len(self):
                raise ValueError("Length of array must match DataTable's length")
        elif isinstance(value, DataTable):
            if value.shape[1] != 1:
                raise ValueError("Setting DataTable must be of a single column")
            if len(value) != len(self):
                raise ValueError("Setting DataTable must have same length as current DataTable")
            value = next(iter(value._data.values()))
        elif isinstance(value, (int, bool, str, float)):
            value = np.repeat(value, len(self))
        else:
            raise TypeError("Value must be either of: DataTable, array, int, bool, str, float")

        if value.dtype.kind == 'U':
            value = value.astype('object')

        self._data[key] = value
