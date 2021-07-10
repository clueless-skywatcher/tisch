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

    



