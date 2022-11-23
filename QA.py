import pandas as pd

from Formats import Formats
fmt = Formats()

class QARawvsFinal:

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

    def obtain_final_sum_nongrouped(self, col_to_check: str): 

        #obtain sum
        sum_final = self.dffinal[col_to_check].sum()

        #return
        return sum_final

    def compare_starting_vs_final_nongrouped(self, raw_filter_criteria, starting_col_name: str, ending_col_name: str, is_currency: bool, threshold=2):

        #obtain starting sum
        starting_sum = self.obtain_starting_sum_nongrouped(raw_filter_criteria=raw_filter_criteria, col_to_check=starting_col_name)
        starting_sum = round(starting_sum, 0)

        #obtain final sum
        final_sum = self.obtain_final_sum_nongrouped(col_to_check=ending_col_name)
        final_sum = round(final_sum, 0)

        #obtain variance
        variance = final_sum - starting_sum

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
