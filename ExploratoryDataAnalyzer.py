#Import Classes

#data handles
import numpy as np
import pandas as pd

#visualizers
import matplotlib.pyplot as plt
import seaborn as sns

import Stats

## Create Class
class ExploratoryDataAnalyzer:

    def __init__(self, df: pd.DataFrame(), predictors: list, target: str):
        self.df = df
        self.target = target
        self.predictors = predictors
        self.graph_size_medium = (14, 5)

    def show_duplicates(self):
        duplicates = self.df[self.df.duplicated()]
        return duplicates
    
    def show_proportions(self, df, col: str, verbose=False):
        '''
        shows proportion (percent and cumulative percent) for selected dataframe and column
        '''
        
        dfProportion = pd.DataFrame()
        dfProportion['Count'] = self.df[col].value_counts(normalize=False)
        dfProportion['Percent'] = self.df[col].value_counts(normalize=True)
        dfProportion['Cumulative %'] = dfProportion['Percent'].cumsum()

        if verbose==True:
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(dfProportion)
            pd.set_option('display.float_format', lambda x: '%.0f' % x)

        return dfProportion

    def show_target_grouping_statistics(self):
        '''
        Purpose \n
        Shows means, medians grouped by the target predictor.
        '''
        
        means = self.df.groupby(self.target).mean().round(0).reset_index(drop=True)
        medians = self.df.groupby(self.target).median().reset_index(drop=True)

        print(f'mean stats: \n {means} \n')
        print(f'median stats: \n {medians} \n')

    def plot_distribution(self, of_columns: list):
        for col in of_columns:
            #create plot of distributions for each columns
            sns.histplot(data=self.df, x=col, bins=100)
            plt.title('Histogram of ' + col, size=35)
            plt.xlabel(col, size=20)
            plt.ylabel('Frequency', size=20)
            plt.xticks(size=20)
            plt.yticks(size=20)
            plt.show()

            #also show statistics just below each distribution for quantitative analysis
            print(col, 'Statistics')
            
            #show skew
            skew = self.df[col].skew()
            print('skew:', str(round(skew,2)))

            #show kurtosis
            kurtosis = self.df[col].kurtosis()
            print('kurtosis:', str(round(kurtosis,2)))
            
            #show mean/median/mode
            mean = round(self.df[col].mean(), 1)
            median = round(self.df[col].median(), 1)
            mode = list(round(self.df[col].mode(), 1))
            min = round(self.df[col].min(), 2)
            max = round(self.df[col].max(), 2)
            print('mean:', mean, '|| median:', median, '|| mode:',mode)
            print('min:',min, '|| max:',max)
            print('\n')
    
    def plot_categorical_outlier_impact_on_target_variable(self):
        categorical_cols = self.df.select_dtypes(include='category').columns.to_list()

        if len(categorical_cols)<1:
            print('must have at least 1 categorical column.')
            exit

        for col in categorical_cols:
            _ = plt.figure()
            _ = sns.boxplot(x=col, y=self.target, data=self.df)
            _ = plt.title('Categorical Impact on HighScore', size=35)
            _ = plt.xticks(rotation=45, size=18)
            _ = plt.yticks(size=18)
            _ = plt.xlabel('Categories', size=20)
            _ = plt.ylabel(self.target, size=20)
            plt.show()
            plt.clf()
            plt.close()

    def plot_blanks_per_column(self):

        df_blanks = self.df.isnull().sum().rename('Count')
        df_blanks = df_blanks.reset_index()
        df_blanks = df_blanks[df_blanks['Count']>0]

        if len(df_blanks)<1:
            print('No blanks in dataset')
        else:
            sns.barplot(data=df_blanks, x='index', y='Count')
            plt.title('Count of Blanks Per Column')
            plt.xlabel('Columns')
            plt.ylabel('Count Blanks')
            plt.show()

    def plot_target_balance(self, graph_size=None):
        '''
        purpose
        plots counts per target classification to show how disproportionate it is

        inputs
        graph_size: input as tuple (x, y) where x = width, y=height
        '''

        #set graph size
        if graph_size==None:
            graph_size=self.graph_size_medium
        plt.rcParams["figure.figsize"] = graph_size

        #plot target variable
        _ = sns.countplot(data=self.df, x=self.target, palette='hls', order=self.df[self.target].value_counts().index)
        plt.title('Prediction Balance')
        plt.show()

    def plot_correlation_matrix(self):
        '''
        Purpose
            Plots correlation matrix to show correlative power of each numeric column against all other numeric columns.

        Inputs
            None

        Outputs
            Correlation Plot
        '''
        
        # Compute the correlation matrix
        corr = self.df.corr(method = 'pearson')

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask 
        sns.heatmap(corr, 
                    mask = mask, 
                    cmap = cmap, 
                    vmax = 1,                                      # Set scale min value
                    vmin = -1,                                     # Set scale min value
                    center = 0,                                    # Set scale min value
                    square = True,                                 # Ensure perfect squares
                    linewidths = 1.5,                              # Set linewidth between squares
                    cbar_kws = {"shrink": .9},                     # Set size of color bar
                    annot = True,                                   # Include values within squares
                    fmt = '.2f',
                );

        plt.xticks(rotation=45)                                    # Rotate x labels
        plt.yticks(rotation=45)                                    # Rotate y labels
        plt.title('Correlation Plot', size=30, y=1.01);   # Set plot title and position
        plt.show()

    def plot_n_lowest_and_highest_category_against_target(self, category: str, n: int):
        ct_scorevscategory = pd.crosstab(self.df[category], self.df[self.target]).reset_index().set_index(category)

        ct_scorevscategory['variance'] = ct_scorevscategory[1] - ct_scorevscategory[0]
        most_popular = ct_scorevscategory.sort_values('variance', ascending=False).drop(columns='variance')
        most_unpopular = ct_scorevscategory.sort_values('variance', ascending=True).drop(columns='variance')                                                                        

        def plot_bar_graph(df, title: str, n: int):
            _ = df.head(n).plot(kind='bar')
            _ = plt.rcParams["figure.figsize"] = (10, 10)
            _ = plt.title('{} {} Desirable Categories'.format(n, title), size=35)
            _ = plt.ylabel('Count', size=18)
            _ = plt.xlabel('Category', size=20)
            _ = plt.xticks(rotation=45, size=20)
            _ = plt.yticks(size=20)
            _ = plt.show()
            _ = plt.clf()
            _ = plt.close()

        plot_bar_graph(most_popular, title='Most', n=n)
        plot_bar_graph(most_unpopular, title='Least', n=n)

    def multi_ecdf(self, series_to_compute: list, title: str, xlabel: str, legend_title: str):
        for series in series_to_compute:
            x, y = Stats().empirical_cumulative_distribution(series)
            _ = plt.plot(x, y, marker='.', linestyle='none', alpha=.6, label=series.name)
        _ = plt.xlabel(xlabel)
        _ = plt.ylabel('ECDF')
        _ = plt.ticklabel_format(style='plain', axis='x')
        _ = plt.title(title)
        plt.margins(0.02)
        plt.legend(loc='lower right', fancybox=True, facecolor='white', shadow=True, title=legend_title)
        plt.show()