import pandas as pd

from TestDataFrames import dfsimple

from Formats import Formats
fmt = Formats()

from IPython.display import display, HTML

from types import SimpleNamespace

class QARawvsFinal:
    '''
    Tests if raw sums equals final sum.
    '''

    def __init__(self, dfraw: pd.DataFrame, dffinal: pd.DataFrame):
        self.dfraw = dfraw
        self.dffinal = dffinal

    def obtain_starting_sum_nongrouped(self, raw_filter_criteria, col_to_check: str):
        
        #filter initial talbe if necessary
        if len(raw_filter_criteria)!=0:
            dfrawfiltered = self.dfraw[raw_filter_criteria]
        else:
            dfrawfiltered = self.dfraw

        #obtain sum
        sum_start = dfrawfiltered[col_to_check].sum()

        #return
        return sum_start

    def obtain_expectedvariance_sum_nongrouped(self, filter_criteria, col_to_check: str):
        
        #filter initial talbe if necessary
        if len(filter_criteria)!=0:
            dffinal = self.dffinal[filter_criteria]
        else:
            dffinal = self.dffinal

        #obtain sum
        sum_expectedvariance = dffinal[col_to_check].sum()

        #return
        return sum_expectedvariance
         
    def obtain_final_sum_nongrouped(self, col_to_check: str): 

        #obtain sum
        sum_final = self.dffinal[col_to_check].sum()

        #return
        return sum_final

    def compare_starting_vs_final_nongrouped(self, raw_filter_criteria, starting_col_name: str, ending_col_name: str, is_currency: bool, threshold=2, expectedvariance=False, expectedvariance_filter_criteria=''):
        '''
        Purpose
            Compares starting vs final sum, including any expected variances.

        Parameters
            raw_filter_criteria: (df.column != 'This')
                If needed, add filtering criteria for raw starting dataframe
            starting_column_name: str
                The name of the column in the starting raw data that you want to sum.
            ending_column_name: str
                The name of the column in the ending final data that you want to sum.
            is_currency: bool
                True if you're comparing $s, so that the outputs are formatted accordingly.
            threshold: int
                The variance you are ok with ignoring, due to rounding errors etc. Default is 2.
            expectedvariance: bool
                If you expect a variance between the raw and final data (such as due to manually adding $s between the two data sets) then put this as True. Default is False.
            expectedvariance_filter_criteria: (df[df['col']=''])
                If needed, add filtering criteria for final dataframe to get to the manually added variances

        Requirements
            If you added data between the raw and final data, then you must be able to filter to that data in the final dataset. For example, have a column that says ['Manually Added'] = True and make that your expectedvariance_filter_criteria. The expected variances are derived from the final dataframe.

        '''

        #obtain starting sum
        starting_sum = self.obtain_starting_sum_nongrouped(raw_filter_criteria=raw_filter_criteria, col_to_check=starting_col_name)
        starting_sum = round(starting_sum, 0)

        #obtain expected variance
        if expectedvariance == True:
            expectedvariance_sum = self.obtain_expectedvariance_sum_nongrouped(filter_criteria=expectedvariance_filter_criteria, col_to_check=ending_col_name)
            expectedvariance_sum = round(expectedvariance_sum, 0) 
        else:
            expectedvariance_sum = 0

        #obtain final sum
        final_sum = self.obtain_final_sum_nongrouped(col_to_check=ending_col_name)
        final_sum = round(final_sum, 0)

        #obtain variance
        variance = final_sum - (starting_sum + expectedvariance_sum)

        #determine equivalence
        if abs(variance)>=threshold:
            equivalent = False
        else:
            equivalent = True

        #format sums for printing
        if is_currency:
            starting_sum = fmt.currency(starting_sum)
            final_sum = fmt.currency(final_sum)
            variance = fmt.currency(variance)

        #result if equivalent
        if equivalent==True:
            result = f'|✓| Starting sum equal to final sum: {final_sum}'
            return print(result)

        #result if not equivalent
        if equivalent==False:
            result = f'|X| Starting sum on [{starting_col_name}] not equal to final sum on [{ending_col_name}]:'
            start = f'Starting: {starting_sum}'
            final = f'Final: {final_sum}'
            variance = f'Variance: {variance}'
            return print(result, start, final, variance, '', sep='\n')

class QAOneDataFrame:
    '''
    Purpose
    -----------
        Perform tests related to a single dataframe

    Parameters
    ------------
        df: pd.DataFrame
            The dataframe you want to test
    '''

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_percentage_equals_to_100_percent(self, column_to_groupby: str, column_with_percentage: str):
        '''
        Purpose
        -----------
            Tests if a target column is fully distributed.

        Parameters
        ------------
            column_to_groupby: str
                The column which should be grouped such that the corresponding percentage column totals to 100% when grouped by this item.
            column_with_percentage: str
                The column with the percentages.
        '''

        dfgrouped = self.df.groupby(by=column_to_groupby)[column_with_percentage].sum().reset_index()
        dfnotfullydistributed = dfgrouped[dfgrouped[column_with_percentage].round(2)!=1]

        if len(dfnotfullydistributed)!=0:
            print('|X| The following items have less than 100% distribution:')
            display(HTML(dfnotfullydistributed.to_html()))
        else:
            print('|✓| All items were 100% distributed.')

    def check_equivalence(self, all_columns_to_check: list, columns_to_combine: str, column_quantified_columns_should_equal: str):
        '''
        Purpose
        ------------
            Checks equivalence of given columns.

        Inputs
        -----------
            all_columns_to_check: list
                All the columns you want to check as strings in a list.
            columns_to_combine: str
                The columns you want to combine prior to checking equality with some other column, expressed as a single string. Each column must be started with "n."
                Ex: 'n.ytd_actuals - n.ytd_plan'
            column_quantified_columns_should_equal: str
                The column you want to check equality with, expressed as a string, prepended with "n."
                Ex: 'n.ytd_varaince'
        '''
        
        d_sums = {}
        for col in all_columns_to_check:
            #name sum
            sum_name = col
            
            #obtain sum
            col = self.df[col].sum()
            col = round(col, 2)

            #store sum
            d_sums[sum_name] = col

        n = SimpleNamespace(**d_sums)
        
        #obtain result of columns to combine
        result = eval(columns_to_combine)
        result = round(result, 2)

        #obtain expected result, the equality column
        expected_result = eval(column_quantified_columns_should_equal)
        expected_result = round(expected_result, 2)

        if result==expected_result:
            print(f'|✓| The following logical relationship between the columns is true: {columns_to_combine} == {column_quantified_columns_should_equal}')
        else:
            print(f'|X| The following logical relationship between the columns is NOT true: {columns_to_combine} == {column_quantified_columns_should_equal}')
            print('columns to combine equals: ', result)
            print('expected value was: ', expected_result)

class QATwoDataFrames:

    '''
    Use to compare various factors across two dataframes.

    Inputs
    ------------
        df1: pd.DataFrame
            The first dataframe you want to run QA on.
        df2: pd.DataFrame
            The second dataframe you want to run QA on.
    '''

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):
        self.df1 = df1
        self.df2 = df2
    
    def compare_valuecolumn_acrossgroups(self, df1_groupby_columns: list[str], df2_groupby_columns: list[str], df1_value_column: str, df2_value_column: str):

        '''
        Combines two dataframes and compares single value column across dataframes.

        Inputs
        ------------
            df1_groupby_column: list[str]
                The name of the columns you want to group by in the classes first dataframe.
            df2_groupby_column: list[str]
                The name of the columns you want to group by in the classes second dataframe. MUST be in the same order as the first groupby columns for when column names are standardized across dataframes.
            df1_value_column: str
                The name of the value column in the classes first dataframe.
            df2_value_column: str
                The name of the value column in the classes second dataframe.
        
        '''

        def standardize_value_column_name():
            store_df_name = self.df1.name
            self.df1 = self.df1.rename(columns={df1_value_column: df2_value_column})
            self.df1.name = store_df_name #name gets removed during column rename

        def standardize_group_by_column_name():
            i=0
            
            for col in df1_groupby_columns:
                #store name
                store_df_name = self.df1.name
                
                #rename columns in df1 to match df2 columns
                rename_to = df2_groupby_columns[i]
                self.df1 = self.df1.rename(columns={col: rename_to})
                i += 1

                #add name back
                self.df1.name = store_df_name #name gets removed during column rename

        def add_dataframe_source_column():
            self.df1['df_source'] = self.df1.name
            self.df2['df_source'] = self.df2.name

        def combine_dataframes():
            self.dfcombined = pd.concat([self.df1, self.df2])

        def round_value_column():
            self.dfcombined[df2_value_column] = self.dfcombined[df2_value_column].round(2)
        
        def groupby_and_compare():
            columns_to_groupby = df2_groupby_columns + ['df_source']
            self.dfcombined = self.dfcombined.groupby(columns_to_groupby)[df2_value_column].sum().reset_index()

        def pivot_source():
            self.dfcombined = pd.pivot_table(data=self.dfcombined, index=df2_groupby_columns, columns='df_source', values=df2_value_column).reset_index()
            self.dfcombined.columns.name = None

        def add_variance_column():
            self.dfcombined['variance'] = self.dfcombined[self.df2.name] - self.dfcombined[self.df1.name]

        def run_pipeline():
            standardize_value_column_name()
            standardize_group_by_column_name()
            add_dataframe_source_column()
            combine_dataframes()
            round_value_column()
            groupby_and_compare()
            pivot_source()
            add_variance_column()

            return self.dfcombined

        run_pipeline()
        return self.dfcombined
        

if __name__=='__main__':
    def test_qa_rawvsfinal(df: pd.DataFrame):

        #change df
        dffinal = dfsimple.copy()
        dffinal = dffinal[dffinal.text!='This']

        #write filter for class
        #self.dfraw[raw_filter_criteria]
        filter = (df.text != 'This')
        
        QARawvsFinal(dfraw=df, dffinal=dffinal).compare_starting_vs_final_nongrouped(raw_filter_criteria=filter, starting_col_name='sum_numbers', ending_col_name='sum_numbers', is_currency=False)
    #test_qa_rawvsfinal(df=dfsimple)

    def test_qatwodataframes():
        df1 = pd.DataFrame({
        'Column1': ['A', 'B', 'C', 'D'],
        'ColumnA': ['Bob', 'Sally', 'Rex', 'John'],
        'Column3': ['This', 'Should', 'Be', 'Ignored'],
        'ValueCol_oldname': [200, 300, 400, 500]
        })

        df2 = pd.DataFrame({
            'Column1': ['A', 'B', 'C', 'D'],
            'Column2': ['Bob', 'Sally', 'Rex', 'John'],
            'Column3': ['This', 'Should', 'Be', 'Ignored'],
            'ValueCol': [200, 350, 450, 500]
        })

        df1.name = 'df1'
        df2.name = 'df2'

        dftest = QATwoDataFrames(df1=df1, df2=df2).compare_valuecolumn_acrossgroups(df1_groupby_columns=['Column1', 'ColumnA'], df2_groupby_columns=['Column1', 'Column2'], df1_value_column='ValueCol_oldname', df2_value_column='ValueCol')
        print(dftest)

    test_qatwodataframes()
