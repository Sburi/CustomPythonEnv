import numpy as np
import pandas as pd

dfsimple = pd.DataFrame({
    'sequential_numbers': [1, 2, 3, 4, 5],
    'text': ['This', 'Is', 'Always', 'A', 'Test'],
    'sum_numbers': [5, 10, 15, 20, 25],
})


dfwithblanks = pd.DataFrame({
    'sequential_numbers': [1, 2, 3, 4, 5],
    'text': ['This', '', 'A', np.nan, 'Test'],
    'sum_numbers': [5, 10, 15, 20, 25],
})