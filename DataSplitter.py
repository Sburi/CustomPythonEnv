# Load Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Create Class
class DataSplitter:
    '''
    Splits data into X and Y trains prior to data preprocessing and model ingestion.
    '''
    
    def __init__(self, df: pd.DataFrame(), X: list, y: str):
        self.df = df
        self.X = X
        self.y = y
    
    def train_test_split(self, test_size:.3):
        '''
        performs a simple train_test_split to the

        inputs
        test_size: what percent of data should be retained for testing
        '''
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.X], self.df[self.y], test_size=test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split_convert_y_to_dummies(self, test_size:.3):
        '''
        splits test and train sets while converting the target to dummies

        inputs
        test_size: what percent of data should be retained for testing
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.X], pd.get_dummies(self.df[self.y]), test_size=test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_test_split_remove_outliers(self, test_size:.3, contamination=.2, bootstrap=False):
        '''
        Offers one method of outlier removal during train/test splitting

        inputs
        contamination: A frac between 0-1, representing the expected outliers to real data ratio
        bootstrap: Change to true if you want to use the bootstrap method
        test_size: what percent of data should be retained for testing
        '''

        # Extract the data without column names from the dataframe
        used_cols = self.X + [self.y]
        data = self.df[used_cols].values

        # Split the data into X/y train/test data
        X, y = data[:, :-1], data[:, -1]
        print('X: ', X, '\n', 'y: ', y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=1)

        # Set up the Isolation Forest classifier that will detect the outliers
        i_forest = IsolationForest(contamination=contamination, bootstrap=bootstrap)
        is_inlier = i_forest.fit_predict(self.X_train)    # +1 if inlier, -1 if outlier

        # Finally, we will select the rows without outliers
        mask = is_inlier != -1
        # and remove these from the train data
        self.X_train, self.y_train = self.X_train[mask, :], self.y_train[mask]  

        return self.X_train, self.X_test, self.y_train, self.y_test      

    def label_and_recombine_train_X_and_y(self):
        self.df_train = pd.concat([self.X_train, self.y_train], axis=1)
        return self.df_train

    def resplit_x_and_y_after_preprocessing(self, df_train: pd.DataFrame()):
        self.X_train = df_train[self.X]
        self.y_train = df_train[self.y]
        return self.X_train, self.y_train
    
    def check_shape(self):
        '''
        checks shape of split data
        '''
        xshape_train=self.X_train.shape
        xshape_test=self.X_test.shape
        yshape_train=self.y_train.shape
        yshape_test=self.y_test.shape
        print(f'X Train Shape: {xshape_train}')
        print(f'Y Train Shape: {yshape_train}')
        print(f'X Test Shape: {xshape_test}')
        print(f'Y Test Shape: {yshape_test}')



