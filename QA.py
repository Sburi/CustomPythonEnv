import pandas as pd

from TestDataFrames import dfsimple

from Formats import Formats
fmt = Formats()

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


    # def compare_dataframes(self, col):
    #     #round
    #     dfmerge[col_mainforecasttable] = dfmerge[col_mainforecasttable].round(0)
    #     dfmerge[col_dlforecasttable] = dfmerge[col_dlforecasttable].round(0)

    #     #create variance column
    #     dfmerge[col_variance] = dfmerge[col_mainforecasttable] - dfmerge[col_dlforecasttable]
    #     dfmerge[col_variance] = dfmerge[col_variance].round(0)
    #     dfmerge[col_variance] = dfmerge[col_variance].fillna(0)

    #     #anaylze
    #     dfmerge.loc[
    #         dfmerge[col_variance].round(0)!=0,
    #     'Equal'] = 'FALSE'

    #     #filter
    #     dfmerge = dfmerge[dfmerge['Equal'] == 'FALSE']
    #     dfmerge = dfmerge.drop(columns=['Equal'])

    #     #show results
    #     if len(dfmerge)==0:
    #         print('|✓| Staying in pre-existing org matches between main and direct labor tables.')
    #     else:
    #         print('|X| Staying in pre-existing org does not between main and direct labor tables.\n')
        
    #         #format columns
    #         columns = [col_mainforecasttable, col_dlforecasttable]
    #         for col in columns:
    #             dfmerge[col] = dfmerge[col].fillna(0)
    #             dfmerge[col] = dfmerge[col].apply(lambda x: f"${x*1:,.0f}")
    #         print(dfmerge)


if __name__=='__main__':
    def test_qa_rawvsfinal(df: pd.DataFrame):

        #change df
        dffinal = dfsimple.copy()
        dffinal = dffinal[dffinal.text!='This']

        #write filter for class
        #self.dfraw[raw_filter_criteria]
        filter = (df.text != 'This')
        
        QARawvsFinal(dfraw=df, dffinal=dffinal).compare_starting_vs_final_nongrouped(raw_filter_criteria=filter, starting_col_name='sum_numbers', ending_col_name='sum_numbers', is_currency=False)

    test_qa_rawvsfinal(df=dfsimple)