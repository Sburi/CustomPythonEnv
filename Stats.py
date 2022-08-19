import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Formats

class Stats:
    def empirical_cumulative_distribution(self, series: pd.Series):
        n = len(series)
        x = np.sort(series)
        y = np.arange(1, n+1) / n
        return x, y

    def correlational_coefficient(self, series1: pd.Series, series2: pd.Series):
        correlational_coefficient = np.corrcoef(series1, series2)[0,1]
        print(correlational_coefficient)

    def check_for_normal_distribution(self, series_to_check: pd.Series()):
        #obtain mean and std
        mean = np.mean(series_to_check)
        std_deviation = np.std(series_to_check)

        # Sample out of a normal distribution
        samples = np.random.normal(mean, std_deviation, 10000)

        # Get the CDF of the samples and of the data
        x_theor, y_theor = Stats.ecdf(samples)
        x_actual, y_actual = Stats.ecdf(series_to_check)

        # Plot the CDFs and show the plot
        _ = plt.plot(x_theor, y_theor)
        _ = plt.plot(x_actual, y_actual, marker='.', linestyle='none')
        _ = plt.xlabel(series_to_check.name)
        _ = plt.ylabel('CDF')
        _ = plt.title('Check for Normal Distribution')
        plt.show()

    def univariate(df, columns_to_test: list):   
        df_results = pd.DataFrame()
        for col in columns_to_test:
            
            #obtain stats
            median = np.median(df[col]); median = Formats.currency(median)
            mean = np.mean(df[col]); mean = Formats.currency(mean)
            variance = np.var(df[col]); variance = Formats.currency(variance)
            standard_deviation = np.std(df[col]); standard_deviation= Formats.currency(standard_deviation)
            
            #create df
            df_to_append = pd.DataFrame.from_dict({'column': col, 'median': [median], 'mean':[mean], 'standard deviation': standard_deviation})
            df_results = pd.concat([df_results, df_to_append], ignore_index=True)
        print(df_results)

    def bivariate(series1: str, series2: str):
        
        name = series1.name + ' vs ' + series2.name
        print(name)
        covariance = np.cov(series1, series2)[0,1]
        correlational_coefficient = np.corrcoef(series1, series2)
        print(correlational_coefficient)
        print(correlational_coefficient[0,1])
        dct = {'covariance': [covariance], 'correlational coefficient': [correlational_coefficient]}

        df_results = pd.DataFrame.from_dict(dct)

        print(df_results)