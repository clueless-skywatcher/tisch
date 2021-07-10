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

        _ = df.dtypes