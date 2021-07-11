import random
import tisch
import pytest
import numpy as np

class TestDataTableCreate:
    
    def test_input_type(self):
        with pytest.raises(TypeError):
            tisch.DataTable([1, 2, 3])
        with pytest.raises(TypeError):
            tisch.DataTable({1: 2, '2': 3})
        with pytest.raises(TypeError):
            tisch.DataTable({"1": np.array([1]), "2": 3})
        with pytest.raises(ValueError):
            tisch.DataTable({"a": np.array([1]), "b": np.array([[1]])})
        
        tisch.DataTable({
            "a": np.array([1, 2, 3, 4]),
            "b": np.array([5, 6, 7, 8])
        })
    
    def test_array_length(self):
        with pytest.raises(ValueError):
            arr1 = np.array([1, 2, 3])
            arr2 = np.array([4, 5])
            tisch.DataTable({
                "a": arr1,
                "b": arr2
            })
        
        tisch.DataTable({
            "a": np.array([1, 2, 3]),
            "b": np.array([3, 4, 5])
        })

    def test_unicode_to_object(self):
        tisch.DataTable({
            "a": np.array(['aa', 'bb', 'cc']),
            "b": np.array(['aa', 'cc', 'bb'])
        })

    def test_set_cols(self):
        df = tisch.DataTable({
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5, 6])
        })
        with pytest.raises(TypeError):
            df.columns = 5
        with pytest.raises(ValueError):
            df.columns = ['1', '2', '3']
        with pytest.raises(TypeError):
            df.columns = ['1', 2]
        with pytest.raises(ValueError):
            df.columns = ['1', '1']
        df.columns = ['c', 'd']

    def test_shape(self):
        df = tisch.DataTable({
            'a': np.array([1, 2, 3]),
            'b': np.array([4, 5, 6])
        })

        _ = df.shape

    def test_values(self):
        df = tisch.DataTable({
            "a": np.array([random.random() for _ in range(10000)]), 
            "b": np.array([random.random() for _ in range(10000)])
        })

        _ = df.values

    def test_dtypes(self):
        df = tisch.DataTable({
            "a": np.array([random.random() for _ in range(10000)]), 
            "b": np.array([random.random() for _ in range(10000)])
        })

class TestDataTableSelect:
    def test_get_item(self):
        df = tisch.DataTable({
            "a": np.array([random.random() for _ in range(10000)]), 
            "b": np.array([random.random() for _ in range(10000)])
        })

    def test_bool_select(self):
        df_tisch = tisch.DataTable({
            "a": np.array([True, False, True, True]), 
            "b": np.array([0.5, 0.32, 4.1, 5.2]), 
            "c": np.array([1, 2, 3, 4])
        })

        with pytest.raises(ValueError):
            df_bool = df_tisch[['a', 'b']]
            _ = df_tisch[df_bool]

        with pytest.raises(ValueError):
            df_bool = df_tisch['b']
            _ = df_tisch[df_bool]

        df_bool = df_tisch['a']
        _ = df_tisch[df_bool]

    def test_wrong_selection_type(self):
        with pytest.raises(TypeError):
            df_tisch = tisch.DataTable({
                "a": np.array([True, False, True, True]), 
                "b": np.array([0.5, 0.32, 4.1, 5.2]), 
                "c": np.array([1, 2, 3, 4])
            })

            _ = df_tisch[True]

    def test_tuple_selection(self):
        with pytest.raises(TypeError):
            df_tisch = tisch.DataTable({
                "a": np.array([True, False, True, True]), 
                "b": np.array([0.5, 0.32, 4.1, 5.2]), 
                "c": np.array([1, 2, 3, 4])
            })

            _ = df_tisch["a", "b", "c"]
        

