import string

import numpy as np
import pandas as pd

from scipy import stats

import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import StandardScaler

# class Columns:
#     def __init__(self):
#         self.col_name = col_name

class Preprocessor:
    def __init__(self, df):
        self.df = df
        Columns.__init__(self)

    def convert_to_categorical_dtype(self, columns_to_convert: list):
        for col in columns_to_convert:
            self.df[col] = self.df[col].astype('category')

    def remove_outliers_based_on_stddev(self, cols_to_check: list, std_deviations_to_keep: int, verbose=False):
        
        #obtain initial shape for verbose
        initial_shape = self.df.shape

        #removes any rows for the specific columns where the values are <= the specific number of standard deviations
        for col in cols_to_check:
            df_revised = self.df[(np.abs(stats.zscore(self.df[col])) <= std_deviations_to_keep)]
    
        #return dataframe
        self.df = df_revised

        #print results
        ending_shape = self.df.shape
        if verbose==True:
            print(f'initial shape: {initial_shape}')
            print(f'ending shape: {ending_shape}')
            print(f'rows removed: {initial_shape - ending_shape}')

    def remove_outliers_based_on_quantiles(self, cols: list, upper_bound=.99, lower_bound=.01, verbose=False):
        
        #obtain initial shape for verbose
        initial_shape = self.df.shape

        #remove outliers beyond specific bounds
        for col in cols: 
            lower_qt = self.df[col].quantile(lower_bound)
            upper_qt = self.df[col].quantile(upper_bound)
            self.df = self.df[(self.df[col]>lower_qt) & (self.df[col]<upper_qt)]
        
        #print results
        ending_shape = self.df.shape
        if verbose==True:
            print(f'initial shape: {initial_shape}')
            print(f'ending shape: {ending_shape}')
            print(f'rows removed: {initial_shape - ending_shape}')
        
    def drop_missing_categories(self):
        self.df = self.df[self.df[self.col_category].notnull()]

    def clean_text_column(self, column_toclean: str, column_new: str):
        def clean_text(self, text, stem='None'):
            final_string = ''

            #format
            text = text.lower()

            #remove line breaks
            text = re.sub(r'\n', ' ', text)

            #remove punctuations
            table = str.maketrans('', '', string.punctuation)
            text = ' '.join(text.translate(table).split())
            
            #remove stop words
            text = text.split()
            useless_words = nltk.corpus.stopwords.words('english')
            useless_words = useless_words + ['hi', 'im']
            text_filtered = [word for word in text if not word in useless_words]

            #stem or lemmatize
            if stem == 'Stem':
                stemmer = PorterStemmer()
                text_stemmed = [stemmer.stem(y) for y in text_filtered]
            elif stem == 'Lem':
                lem = WordNetLemmatizer()
                text_stemmed = [lem.lemmatize(y) for y in text_filtered]
            else:
                text_stemmed = text_filtered

            final_string = ' '.join(text_stemmed)

            return final_string

        self.df[column_new] = self.df[column_toclean].apply(lambda x: clean_text(self, text=x, stem='Stem'))
    
    def scale_data(self, numeric_columns: list):
        scaler = StandardScaler()
        self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
      
    def normalize_values_based_on_dict(self, col_to_standardize: str, dictionary_with_conversions: dict, print_conversions=False, revised_col='', classify_nonspecified_as_other=False):
        '''normalize values based on dict where key = correct, values = list of incorrect instances'''

        # #backup orig
        # dfOrig = df[[col_to_standardize]].copy()
    
        #if no revised column has been provided, overwrite the original column with the revised column
        if revised_col == '':
            revised_col = col_to_standardize

        #normalize vendors
        temp_conversion_col = 'Converted To'
        for k,v in dictionary_with_conversions.items():
            self.df.loc[
                self.df[col_to_standardize].str.lower().isin([i.lower() for i in v]),
            temp_conversion_col] = k
        self.df[temp_conversion_col] = self.df[temp_conversion_col].fillna(self.df[col_to_standardize])

        if classify_nonspecified_as_other==True:
            verified_classifications = list(dictionary_with_conversions.keys())
            self.df.loc[~self.df[self.col_to_standardize].isin(verified_classifications), self.col_to_standardize] = 'Other'

        #print changes
        if print_conversions==True:
            conversions =  self.df.where(self.df[col_to_standardize] != self.df[temp_conversion_col])[[col_to_standardize, temp_conversion_col]]
            conversions.rename(columns={col_to_standardize:'Original Entry'}, inplace=True)
            print('conversions:')
            print(conversions.drop_duplicates().dropna().sort_values('Converted To'))

        #implement change
        self.df[revised_col] = self.df[temp_conversion_col]
        self.df = self.df.drop(temp_conversion_col, axis=1)

    def match_category_counts_to_min_category_count(self, col_to_balance: str):
        min_category = self.df[col_to_balance].value_counts().min()
        unique_categories = self.df[col_to_balance].unique()

        for category in unique_categories:
            total_in_category = len(self.df[self.df[col_to_balance]==category])
            rows_to_remove = total_in_category - min_category

            if rows_to_remove == 0:
                break

            df_to_reduce = self.df[self.df[col_to_balance]==category]
            all_other_categories = self.df[self.df[col_to_balance]!=category]

            df_reduced = df_to_reduce.sample(n=min_category, random_state=1)
            self.df = pd.concat([df_reduced,all_other_categories])

    def series_generator(self, col_to_create_series_from: str, col_to_retain: str):
        '''
        inputs
        col to create series from
        col to retain as series values

        not sure why I made this, need documentation
        '''
        list_of_series = []
        categories = self.df[col_to_create_series_from].unique()
        for category in categories:
            df_filtered = self.df[self.df[col_to_create_series_from]==category]
            series = df_filtered[col_to_retain]
            series.name = category
            list_of_series = list_of_series + [series]
        
        return list_of_series

    def fill_missing_months(self, date_col: str):

        '''
        fills missing months between min and max date with 0 so that a date range becomes continuous.

        reqs:
        data must be in [Date] [Value] columns, no more no less with current programming.

        Before
        1/31/22: 10
        3/31/22: 30

        After
        1/31/22: 10
        2/28/22: 0
        3/31/22: 30
        '''
        #get min, max dates
        min_date = self.df[date_col].min() + pd.tseries.offsets.MonthEnd(-1)
        max_date = self.df[date_col].max() 

        #create list of dates
        full_dates = pd.date_range(min_date, max_date, freq='MS').strftime('%Y-%m-%d').tolist()
        full_dates = pd.to_datetime(full_dates)
        full_dates = full_dates + pd.tseries.offsets.MonthEnd(1)
        df_fulldates = pd.DataFrame({'Date': full_dates})

        #merge target df
        df_combined = df_fulldates.merge(self.df, how='left', left_on='Date', right_on=date_col)
        if date_col != 'Date':
            df_combined = df_combined.drop(columns=date_col)

        #fill nans with 0
        df_combined = df_combined.fillna(0)

        return df_combined

    def standardize_column_names(self, verbose=False):

        '''
        purpose
        enable columns in dataframe to be accessed via dot notation ergo df.columnname
        does this by lower casing each column, replacing spaces with underscores, etc

        input
        dataframe

        output
        dataframe with accessible column names
        '''

        current_cols = self.df.columns

        new_cols = []
        for col in current_cols:
            new_col = col.lower()
            new_col = new_col.replace(' ', '_')
            new_cols.append(new_col)

        self.df.columns = new_cols
        
        if verbose==True:
            print(f'new columns: ', new_cols)

    def add_cumulative_multiple(self, target_cols: list):
        '''
        purpose
        for each column in target_cols, adds a new columns with the target columns cumulative total

        input
        df
        target columns

        outputs
        df
        target columns
        cumulative total per target column
        '''

        for col in target_cols:
            new_col_name = f'{col}_cumulative'
            self.df[new_col_name] = self.df[col].cumsum()




