from os import access
import pandas as pd

##############################################################################################
class BlankRemover:
    
    def __init__(self, df_orig):
        self.df_orig = df_orig
        self.df_refined = df_orig.copy()
        self.cols_removed_due_to_blank_rows = []
        self.cols_removed_due_to_only_one_entry = []
    
    def welcome(self):
        print('')
        print('INITIATING BLANKET PREPROCESSOR')
        print('-------------------------------------------------------------------')

    def remove_blank_rows(self, acceptable_blank_percent=.01):
        print('REMOVING BLANK ROWS')

        #quantifies how much to remove
        count_columns = len(self.df_orig.columns)
        count_max_acceptable_blankcolumns = acceptable_blank_percent * count_columns
        count_max_acceptable_blankcolumns = max(2, count_max_acceptable_blankcolumns)
        #print('total columns: ', count_columns)
        #print('acceptable blank percent: ', acceptable_blank_percent)
        #print('acceptable blank count: ', count_max_acceptable_blankcolumns)

        #removes blanks
        self.df_refined.dropna(axis=0, thresh=count_max_acceptable_blankcolumns, inplace=True)

        #raises changes to user
        removed_content = self.df_orig[~self.df_orig.index.isin(self.df_refined.index)]
        print('rows removed: ', len(self.df_orig) - len(self.df_refined))
        #print('row content: \n', removed_content)

    def remove_blank_columns(self, acceptable_blank_percent = .75):
        print('REMOVING BLANK COLUMNS')

        #setup
        input_df = self.df_refined.copy()

        #quantifies how much to remove
        count_rows = len(input_df)
        count_max_acceptable_blankrows = acceptable_blank_percent * count_rows
        count_max_acceptable_blankrows = max(2, count_max_acceptable_blankrows)
        #print('total rows: ', count_rows)
        #print('acceptable blank percent: ', acceptable_blank_percent)
        #print('acceptable blank count: ', count_max_acceptable_blankrows)

        #removes blanks
        self.df_refined.dropna(axis=1, thresh=count_max_acceptable_blankrows, inplace=True)

        #raise changes to user
        if len(input_df.columns) != len(self.df_refined.columns):
            #print('\nRemoved the below columns, recommend to remove these from any source query: ')
            for col in input_df.columns:
                if col not in self.df_refined.columns:
                    #print(col)
                    self.cols_removed_due_to_blank_rows.append(col)
        else:
            print('no columns removed')

    def remove_single_entry_columns(self):
        print('REMOVING SINGLE ENTRY COLUMNS')

        #setup
        df_input = self.df_refined.copy()
        
        #determine which columns should be dropped
        for col in df_input.columns:
            value_count_size = df_input[col].value_counts().size
            if value_count_size==1:
                self.df_refined.drop(columns=col, inplace=True)

        #raise changes to user
        if len(df_input.columns) != len(self.df_refined.columns):
            #print('removed the following columns due to having only 1 entry throughout, recommend to remove these from incoming query')
            for col in df_input.columns:
                if col not in self.df_refined.columns:
                    #print(col)
                    self.cols_removed_due_to_only_one_entry.append(col)

    def results(self):
        print('')
        print('RESULTS')
        
        print('shape improvements')
        print('Dataframe starting size: ', self.df_orig.shape)
        print('DataFrame ending size: ', self.df_refined.shape)

        print('\nREMOVE THE BELOW COLUMNS FROM INITIAL QUERY IF POSSIBLE')
        print('----------------------------------------------------------')
        print('due to missing data: \n', self.cols_removed_due_to_blank_rows)

        print('\ndue to only one entry: \n', self.cols_removed_due_to_only_one_entry)
        print('----------------------------------------------------------')

    def run_all_methods(self):
        self.welcome()
        self.remove_blank_rows()
        self.remove_blank_columns()
        self.remove_single_entry_columns()
        self.results()

##############################################################################################

if __name__ == "__main__":

    def access_data(path_to_file, filename, filetype, append_to_filename_when_modified = '_Refined'):

        full_file_path = path_to_file + filename + filetype

        if filetype=='.xlsx':
            df_raw = pd.read_excel(full_file_path)
            #path to send refined file
            full_file_path_when_modified = path_to_file + filename + append_to_filename_when_modified + '.xlsx'
            #finalize
            return df_raw, full_file_path_when_modified
        if filetype=='.csv':
            df_raw = pd.read_csv(full_file_path)
            #path to send refined file
            full_file_path_when_modified = path_to_file + filename + append_to_filename_when_modified + '.xlsx'
            #finalize
            return df_raw, full_file_path_when_modified
        else:
            print('please specify filetype = either .xslx or .csv')

    def preprocess_fm_expense_line():
        path_to_file = r'C:\Users\SIB4953\Humana\Common Data Sources\ServiceNow\\' #keep last extra \
        filename = r'fm_expense_line'
        filetype = r'.csv'
        append_to_filename_when_modified = '_Refined'
        df_raw, full_file_path_when_modified = access_data(path_to_file=path_to_file, filename=filename, filetype=filetype, append_to_filename_when_modified=append_to_filename_when_modified)
        blanket_preprocessor = BlanketPreprocessor(df_orig=df_raw)
        blanket_preprocessor.run_all_methods()
        addtl_columns_to_remove = ['sys_created_on', 'source_id', 'sys_created_by', 'u_expenditure', 'u_oracle_unique_id', 'u_ppm_migrated', 'source_table', 'sys_updated_on', 'sys_updated_by', 'sys_mod_count']
        blanket_preprocessor.df_refined.drop(columns=addtl_columns_to_remove, inplace=True)
        print('also remove these columns: ')
        print(addtl_columns_to_remove)
        blanket_preprocessor.df_refined.to_excel(full_file_path_when_modified)
        print('complete')
    #preprocess_fm_expense_line()