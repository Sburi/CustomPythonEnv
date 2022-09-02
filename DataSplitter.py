# Load Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Create Class
class DataSplitter:
    '''
    Contains functions which split and recombine data to prevent data leakage prior to running X and y train and test data through ML models.
    '''
    
    def __init__(self, df: pd.DataFrame(), X: list, y: str, test_size: int):
        self.df = df
        self.X = X
        self.y = y
        self.test_size = test_size
    
    def train_test_split(self):
        '''
        performs a simple train_test_split.

        inputs
        test_size: what percent of data should be retained for testing

        returns
        X_train, X_test, y_train, y_test
        '''
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.X], self.df[self.y], test_size=self.test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split_convert_y_to_dummies(self):
        '''
        splits test and train sets while converting the target to dummies

        inputs
        test_size: what percent of data should be retained for testing

        returns
        X_train, X_test, y_train, y_test
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.X], pd.get_dummies(self.df[self.y]), test_size=self.test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split_remove_outliers(self, contamination=.2, bootstrap=False):
        '''
        Offers one method of outlier removal during train/test splitting

        Inputs
        contamination: A frac between 0-1, representing the expected outliers to real data ratio
        bootstrap: Change to true if you want to use the bootstrap method
        test_size: what percent of data should be retained for testing

        Returns
        X_train, X_test, y_train, y_test

        Notes
        recommend to perform outlier removal after splitting but during preprocessing for more fine tuned control, perhaps using quantiles or standard deviations.
        '''

        # Extract the data without column names from the dataframe
        used_cols = self.X + [self.y]
        data = self.df[used_cols].values

        # Split the data into X/y train/test data
        X, y = data[:, :-1], data[:, -1]
        print('X: ', X, '\n', 'y: ', y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)

        # Set up the Isolation Forest classifier that will detect the outliers
        i_forest = IsolationForest(contamination=contamination, bootstrap=bootstrap)
        is_inlier = i_forest.fit_predict(self.X_train)    # +1 if inlier, -1 if outlier

        # Finally, we will select the rows without outliers
        mask = is_inlier != -1
        
        # and remove these from the train data
        self.X_train, self.y_train = self.X_train[mask, :], self.y_train[mask]  

        return self.X_train, self.X_test, self.y_train, self.y_test      

    def recombine_X_and_y_for_preprocessing(self):
        '''
        Purpose
        Recombines X and y so that preprocessing steps which reduce rows occur on both datasets
        Use "resplit_x_and_y_after_preprocessing()" to re-split X_train and y_train prior to running data through ML models.
        Note that you must be careful not to introduce data leakage if any preprocessing is occurring on the test data. So do not filter rows. If applying any mathmatical changes make sure they are using the #s from the train data, etc.

        Returns
        df_train, df_test
        '''
        
        self.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        self.df_test = pd.concat([self.X_test, self.y_test], axis=1)
        return self.df_train, self.df_test

    def resplit_x_and_y_after_preprocessing(self, df_train: pd.DataFrame(), df_test: pd.DataFrame(), X: list, y: str):
        '''
        Purpose
        Splits train dataset back into x and y parameters after preprocessing.
        Combining X and Y train occurs to keep rows synced when doing row filtering/removal preprocessing.
        X and Y need to be split again prior to running through ML models.

        Inputs
        df_train: Your training dataset
        X: A list of predictors. Must be redefined (rather than using self.attribute) since X may change during processing.
        y: Predicted value. Must be redefined for the same reason as the above reason.

        Returns
        X_train, X_test, y_train, y_test
        '''

        self.X_train = df_train[X]
        self.y_train = df_train[y]
        self.X_test = df_test[X]
        self.y_test = df_test[y]
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def check_shape(self):
        '''
        prints shape of X and y train and test data
        '''
        xshape_train=self.X_train.shape
        xshape_test=self.X_test.shape
        yshape_train=self.y_train.shape
        yshape_test=self.y_test.shape
        print(f'X Train Shape: {xshape_train}')
        print(f'Y Train Shape: {yshape_train}')
        print(f'X Test Shape: {xshape_test}')
        print(f'Y Test Shape: {yshape_test}')

    def check_contents(self, rows=5):
        '''
        prints shape of X and Y train and test sets
        '''

        print(f'X Train \n {self.X_train.head(rows)} \n')
        print(f'y Train \n {self.y_train.head(rows)} \n')
        print(f'X Test \n {self.X_test.head(rows)} \n')
        print(f'y Test \n {self.y_test.head(rows)} \n')




