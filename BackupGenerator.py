import pandas as pd
from datetime import datetime
import os

#custom
from CustomEnv.Standardizer import Standardize

# #configs
# from configparser import ConfigParser
# config = ConfigParser()
# config.read(r'C:\Users\SIB4953\Humana\Documents\My Files\Python\By Project\Costs by Capability\Configs\configs_budgetsbycapability.ini')
# filepaths = config['filepaths']

class BackupGenerator:
    '''
    Purpose
    ------------
    Generates a backup of the target data. Saves the file as the date of backup, adds a column to denote the date of the backup, and then rolls up all prior backups to unified backup.

    Inputs
    -------------
        df: pd.DataFrame
            The dataframe you want to backup.
        filepath: str
            The path to the dataframe you are backing up.
        column_header_conversion_dict: dict
            Optional. A dictionary where the keys are the standardized header names and the values are the non-standard header names you need to unify. Useful if columns have changed over time.
        standardize_column_values_dict: dict
            Optional. A dictionary where the first key is column you want to edit values within, the second key is the correct value, and the values are a list of incorrect values you want to convert to the secondary key. Ex:
                'Column1': {
                    'CorrectValue': ['IncorrectValue1'],
                },
        columnns_to_drop: list
            Optional. Any columns you want to drop.
        columnname_save_date: str
            Optional. If you want to use a different backup name than 'date_of_backup' you may place it here.

    '''
    def __init__(self, df: pd.DataFrame, filepath=None, column_header_conversion_dict=None, standardize_column_values_dict=None, columnns_to_drop=None, columnname_save_date=None):
        #vars set on instantiation
        self.df = df
        self.filepath = filepath
        self.column_header_conversion_dict = column_header_conversion_dict
        self.standardize_column_values_dict = standardize_column_values_dict
        self.columnns_to_drop = columnns_to_drop
        
        #vars set during execution
        self.save_date = ''

        #vars set during instantiation or execution
        if columnname_save_date == None:
            self.columnname_save_date = 'date_of_backup'
        else:
            self.columnname_save_date = columnname_save_date
    
    def obtain_date(self):
        today = datetime.today()
        year = str(today.year)
        month = str(today.month)
        day = str(today.day)
        self.save_date_for_filepath = '_'.join([year, month, day])
        self.save_date_for_column = '/'.join([year, month, day])

    def add_column_save_date(self):
        self.df[self.columnname_save_date] = self.save_date_for_column
    
    def create_backup_folder_if_needed(self):
        #obtain path
        self.savepath = f'{self.filepath}\\AutomaticBackups'
        
        #check if path exists
        exists = os.path.exists(self.savepath)

        #if path doesn't exist, create
        if not exists:
            os.makedirs(self.savepath)
            print(f'Created directory {self.savepath}')

    def export_new_backup(self):
        self.savepathfull = f'{self.savepath}\\{self.save_date_for_filepath}.xlsx'
        self.df.to_excel(self.savepathfull, index=False) 
    
    def combine_all_backups(self):
        
        #obtain all file paths
        allpaths = []
        for file in os.listdir(self.savepath):
            if file!='AllBackupsCombined.xlsx':
                allpaths.append(f'{self.savepath}\\{file}')
        
        #load each file
        alldataframes = []
        for filepath in allpaths:
            #load dataframe
            df = pd.read_excel(filepath)
            
            #standardize column headers
            if self.column_header_conversion_dict!=None:
                standardize = Standardize(df=df, print_conversions=False)
                standardize.standardize_column_headers(conversion_dict=self.column_header_conversion_dict)
                df = standardize.df

            #place into list
            alldataframes.append(df)

        #combine
        self.dfcombined = pd.concat(alldataframes, axis=0)

    def drop_columns(self):
        #drop columns
        if self.columnns_to_drop!=None:
            self.dfcombined = self.dfcombined.drop(columns=self.columnns_to_drop, errors='ignore')
    
    def standardize_column_values(self):
        
        if self.standardize_column_values_dict!=None:
            for k,v in self.standardize_column_values_dict.items():
                column=k
                conversion_dict = v
                standardize = Standardize(df=self.dfcombined, print_conversions=False)
                standardize.standardize_column_values(conversion_dict=conversion_dict, current_col=column)
                self.dfcombined = standardize.df
    
    def export_combined_backups(self):
        self.dfcombined.to_excel(f'{self.savepath}\\AllBackupsCombined.xlsx', index=False)
    
    def run_pipeline(self):
        
        #clean and prepare
        self.obtain_date()
        self.add_column_save_date()

        #ensure backup folder exists
        self.create_backup_folder_if_needed()

        #export
        self.export_new_backup()
        self.combine_all_backups()
        self.drop_columns()
        self.standardize_column_values()
        self.export_combined_backups()

if __name__=='__main__':
    print('check')