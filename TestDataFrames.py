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

dfforbins = pd.DataFrame({
    'Age': [1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 10, 40, 80],
    'Survived': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
})