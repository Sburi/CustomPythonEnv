
from typing import Type
import functools
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import pandas as pd

class IgnoreWarnings:
    def __init__(self):
        pass

    def ignore_warning(self, warning: Type[Warning]):
        """
        Ignore a given warning occurring during method execution.

        Args:
            warning (Warning): warning type to ignore.

        Returns:
            the inner function

        """

        def inner(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category= warning)
                    return func(*args, **kwargs)

            return wrapper

        return inner 

    def ignore_multiple_warnings(self, warning_list: list):
        """
        Ignores specified warnings during method execution.

        Parameters
            warning_list: list
                The warnings you want to ignore, such as FutureWarning

        Returns:
            the inner function

        """

        def inner(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with warnings.catch_warnings():
                    for warning in warning_list:
                        warnings.filterwarnings("ignore", category=warning)
                    return func(*args, **kwargs)

            return wrapper

        return inner 

if __name__ == '__main__':
    dftest = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['Test1', 'Test2', 'Test3'],
    })

    def test_multiple_warnings():
        pass
