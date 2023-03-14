import numpy as np
import pandas as pd
from IPython.display import display, HTML

class MultiDataFrameMethods:
    '''
    Contains methods which can be used on multiple dataframes for the purpose of analysis, comparison, etc.

    Parameters
    ------------
        Dataframes: list (of dataframes)
            A list of the dataframes you want to perform actions upon.

    '''

    def __init__(self, dataframes: list):
        self.dataframes = dataframes

    def compare_dfs(self, print_shared=True, print_unshared=True):
        '''
        Purpose
        ------------
            Compares input list of dataframes to determine which columns are shared vs. unshared between said dataframes.
        
        Parameters
        ------------
            print_shared: boolean
                If true, prints any columns that are shared to screen.
            print_unshared: boolean
                If true, prints any columns that are not shared to screen.

        Outputs
        ------------
            dfComparison with all columns to left, one column to denote shared columns, and one column to denote unshared columns.
        '''

        #ensure more than one df passed
        arg_count = len(self.dataframes)
        if arg_count==1:
            raise ValueError('Must pass more than one dataframe to compare it to other dataframes dummy.')

        #ensure all dataframes have names
        for df in self.dataframes:
            if not(hasattr(df, 'name')):
                raise NameError('All dataframes must have names, assign with df.name = ''')

        #obtain all columns
        cols = []
        for df in self.dataframes:
            cols = cols + list(df.columns)
        cols = list(set(cols))

        #create comparison df
        dfCompare = pd.DataFrame()
        dfCompare['All Columns'] = cols
        
        #check each df for columns
        for df in self.dataframes:
            dfCompare.loc[
                dfCompare['All Columns'].isin(df.columns),
            'Is In ' + df.name] = 'Y'

        #check across all columns
        dfCompare = dfCompare.set_index('All Columns')
        count_cols = len(dfCompare.columns)
        dfCompare['In All'] = ['Y' if x==count_cols else '' for x in np.sum(dfCompare.values=='Y', 1)]

        #if >2 dfs being compared, check if dataset is in multiple dataframes
        if arg_count > 2:
            dfCompare['In Multiple'] = ['Y' if x>1 else x for x in np.sum(dfCompare.values=='Y', 1)]

        #print any observations
        total_cols = len(dfCompare.index)
        #print(total_cols)
        shared_cols = len(dfCompare[dfCompare['In All']=='Y'])
        print('Out of ' + str(total_cols) + ' columns there are ' + str(shared_cols) + ' that are present across all dfs.')

        #print shared cols
        if print_shared==True:
            shared_cols = dfCompare[dfCompare['In All']=='Y'].index.values
            print('shared columns: ', shared_cols)

        #print unshared columns
        if print_unshared==True:
            for col in dfCompare.columns:
                unshared_cols = dfCompare[dfCompare[col].isnull()].index.values
                if len(unshared_cols) > 0:
                    print('['+ str(col).upper() + ']' + ' unshared cols: ', str(list(unshared_cols)))
        
        #reset index
        dfCompare = dfCompare.reset_index()

        #sort
        #dfCompare = dfCompare.sort_values('All Columns')

        return dfCompare

    def show_n_contents(self, rows_to_print: int):
        '''
        Purpose
        ------------
            Prints n contents from each dataframe in class.

        Parameters
        ------------
            rows_to_print: int
                The number of rows you want to print.

        Outputs
        ------------
            Dataframes with n rows printed.
        '''
        
        for df in self.dataframes:
            print(df.head(rows_to_print), '\n')
            display(HTML(df.head(rows_to_print).to_html()))

if __name__ == "__main__":
    # df1 = pd.DataFrame()
    # df1.name='df1'
    # df1['Col A'] = [1,2,3]
    # df1['Col B'] = [1,2,3]
    # df1['Col C'] = [1,2,3]

    # df2 = pd.DataFrame()
    # df2.name='df2'
    # df2['Col A'] = [1,2,3]
    # df2['Col B'] = [1,2,3]

    # test = compare_dfs(dataframes=[df1, df2])
    pass