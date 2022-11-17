
#basics
import itertools

#main data handlers
import numpy as np
import pandas as pd

#custom
from CustomEnv.Decorators import IgnoreWarnings
from CustomEnv.DataSplitter import DataSplitter
iw = IgnoreWarnings()

#models
#from pmdarima import auto_arima
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#model supports
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

#data visualizers
import matplotlib.pyplot as plt
import seaborn as sns

class ReduceForecast:

    def __init__(self, df_full: pd.DataFrame(), col_date: str, col_amount: str, month_frequency: str, test_size=.3):
        self.df = df_full
        self.month_frequency = month_frequency
        self.test_size = test_size
        self.col_date = col_date
        self.col_amount = col_amount

    def select_x_and_y(self):
        self.dfreduced = self.df[[self.col_date, self.col_amount]]
    
    def split_data(self, verbose=False):
        data_splitter = DataSplitter(df=self.dfreduced, test_size=self.test_size, X=None, y=None)
        self.indexes, self.dftrain, self.dftest = data_splitter.train_test_timeseries_splitter()

        if verbose:
            print(f'Train DataFrame: \n {self.dftrain}')
            print(f'\nTest DataFrame: \n {self.dftest}')

    def set_index_as_date(self):
        self.dfreduced = self.dfreduced.set_index(self.col_date).asfreq(self.month_frequency)
        self.dftrain = self.dftrain.set_index(self.col_date).asfreq(self.month_frequency)
        self.dftest = self.dftest.set_index(self.col_date).asfreq(self.month_frequency)  

    def run_pipeline(self, verbose=False):
        self.select_x_and_y()
        self.split_data(verbose=verbose)
        self.set_index_as_date()


class FindPDQ:

    def __init__(self, dfreduced, dftrain, lags: int):
        self.dfreduced = dfreduced
        self.dftrain = dftrain
        self.lags = lags

    def find_p(self):
        '''
        Uses PACF to find q param
        '''
        
        print('Find P')
        print('Look for spikes greater than the threshold, choosing the lowest spike possible to create the simplest model.')
        
        result = seasonal_decompose(self.dfreduced, model='additive', extrapolate_trend='freq')
        plot_pacf(result.seasonal, lags=12, method='ywm')
        plt.show()

    def find_d(self):
        '''
        Uses adfuller test to check for stationarity
        '''
        
        print('\nFind D')
        dfadfuller = adfuller(self.dfreduced)
        adf = dfadfuller[0]
        pvalue = dfadfuller[1]
        critical_value = dfadfuller[4]['5%']
        print(f'adf {adf:.3f} \npvalue {pvalue:.3f} \ncritical value {critical_value:.3f}')
        if (pvalue < 0.05) and (adf < critical_value):
            print('The series is stationary')
        else:
            print('The series is NOT stationary, one or more I in ARIMA will be required')     

    def find_q(self):
        '''
        Uses ACF to find q param
        '''
        
        print('\nFind Q')
        print('Look for spikes greater than the threshold, choosing the lowest spike possible to create the simplest model.')
        result = seasonal_decompose(self.dfreduced, model='additive', extrapolate_trend='freq')
        plot_acf(result.seasonal, lags=self.lags)
        plt.show()

    def show_seasonal_decomposition(self):
        
        print('\nSeasonal Decomposition')
        
        #decompose
        result = seasonal_decompose(self.dfreduced, model='additive', extrapolate_trend='freq')

        #plot, requires 24 observations minimum
        plt.rcParams.update({'figure.figsize':(14,10), 'figure.dpi':120})
        result.plot()
        plt.show()
        pass

    def run_pipeline_manual_search(self):
        self.find_p()
        self.find_d()
        self.find_q()
        self.show_seasonal_decomposition()

    @iw.ignore_warning(UserWarning)
    def grid_search_arima_params(self, max_complexity=2, verbose=False):
        #set list of all possible params to check
        import itertools
        p=d=q=range(0, max_complexity)
        pdq = list(itertools.product(p,d,q))

        #setup df
        aic_list = []
        param_list = []
        
        #loop through each param
        for param in pdq:
            try:
                #setup model
                model_arima = ARIMA(self.dftrain, order=param, freq='M')
                model_arima_fit = model_arima.fit()

                #obtain aic and add to dict
                aic = round(model_arima_fit.aic, 0)
                aic_list.append(aic)
                param_list.append(param)

            except:
                #if param doesn't work on data, check next param
                aic_list.append(aic)
                param_list.append('This param does not work on the data')
                continue

        #get results as df
        dfresults = pd.DataFrame({
            'Param': param_list,
            'AIC': aic_list,
        })

        #get best results
        dfresults = dfresults.sort_values(by='AIC').reset_index(drop=True)
        best_param = dfresults['Param'][0]
        best_aic = dfresults['AIC'][0]
        self.best_param = best_param

        print(f'Best Param: {best_param}, Best AIC: {best_aic}')

        if verbose:
            return dfresults

    @iw.ignore_multiple_warnings([UserWarning, RuntimeWarning])
    def grid_search_sarima_params(self, max_p: int, max_d: int, max_q: int, maxiter=50, freq='M', freq_num=12, verbose=False):
        
        '''
        Input: 
            ts : your time series data
            max_pdq: max iterable you desire to test for optimal params on in Sarima PDQ(p,d,q)
            maxiter : number of iterations, increase if your model isn't converging
            frequency : default='M' for month. Change to suit your time series frequency
                e.g. 'D' for day, 'H' for hour, 'Y' for year. 
            freq_num: The frequency number of the data.
                e.g. 12 for monthly, 365 for daily.

        Example Call
            sarimax_gridsearch(dfreduced, max_p=4, max_d=0, max_q=4, freq='M', freq_num=12)
            
        Return:
            Prints best scores sorted by AIC ascending along with their associated params
        '''

        #setup seasonal
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        pdq = list(itertools.product(p, d, q))

        #setup non_seasonal
        pdqs = [(x[0], x[1], x[2], freq_num) for x in list(itertools.product(p, d, q))]

        # Run a grid search
        ans = []
        for comb in pdq:
            for combs in pdqs:
                try:
                    mod = sm.tsa.statespace.SARIMAX(self.dfreduced,
                                                    order=comb,
                                                    seasonal_order=combs,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False,
                                                    freq=freq)

                    output = mod.fit(maxiter=maxiter) 
                    class_input = f'order={str(comb)}, seasonal_order={str(combs)}'
                    ans.append([comb, combs, output.aic, output.bic, class_input])
                    #print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(comb, combs, output.bic))
                except:
                    if verbose:
                        print(f'the following did not work: {comb}, {combs}')
                    continue
                
        # Convert into dataframe
        dfresults = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic', 'bic', 'class input'])

        #remove useless results
        def remove_useless(df):
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['aic', 'bic'])
            return df
        dfresults = remove_useless(dfresults)

        # Sort and return top 5 combinations
        dfresults = dfresults.sort_values(by=['aic'],ascending=True)
        
        return dfresults

class ArimaForecast:
    '''
    Example Call
    arima_fcst = ArimaForecast(dfreduced=dfreduced, dftrain=dftrain, dftest=dftest, confidence_level=0.95, new_prediction_count=7, frequency='M', arima_params=(0, 1, 1))
    arima_fcst.run_pipeline(verbose=False)
    '''

    def __init__(self, dfreduced, dftrain, dftest, confidence_level: float, new_prediction_count: int, frequency: str, arima_params: tuple):
        self.dfreduced = dfreduced
        self.dftrain = dftrain
        self.dftest = dftest
        self.confidence_level = confidence_level #usually .95
        self.alpha = round(1 - confidence_level,2)
        self.frequency = frequency
        self.new_prediction_count = new_prediction_count
        self.arima_params = arima_params

    def obtain_validation_forecast(self, verbose=False):

        #setup model
        model = ARIMA(self.dftrain, order=self.arima_params, freq=self.frequency)
        self.model_fit_validation = model.fit()
        
        #obtain length of validation
        start_forecast_on = min(self.dftest.index)
        end_forecast_on = max(self.dftest.index)

        #obtain forecast, yhat, conf intervals
        forecast = self.model_fit_validation.get_prediction(start_forecast_on, end_forecast_on)
        self.validation_yhat = forecast.predicted_mean
        self.validation_conf_intervals = forecast.conf_int(self.alpha)

        #join results
        self.df_validation_forecast = pd.merge(self.validation_yhat, self.validation_conf_intervals, how='outer', left_index=True, right_index=True)
        self.df_validation_forecast = self.df_validation_forecast.rename(columns={'predicted_mean':'Validation Forecast - Mean', 'lower sum_all':'Validation Forecast - Low', 'upper sum_all':'Validation Forecast - High'})
        
        #obtain AIC
        aic = self.model_fit_validation.info_criteria('aic')
        print(f'aic: {aic}')

        if verbose:
            print(self.df_validation_forecast.head())

    def obtain_new_forecast(self, verbose=False):

        #setup model
        arima = ARIMA(self.dfreduced, order=self.arima_params, freq=self.frequency)
        self.model = arima.fit()

        #obtain length of validation
        start_forecast_on = max(self.dftest.index)
        end_forecast_on = start_forecast_on + pd.tseries.offsets.MonthEnd(self.new_prediction_count)
        #print(f'start forecast on: {start_forecast_on} \n end forecast on: {end_forecast_on}')

        #obtain forecast, yhat, conf intervals
        forecast = self.model.get_prediction(start_forecast_on, end_forecast_on)
        self.new_yhat = forecast.predicted_mean
        self.new_conf_intervals = forecast.conf_int(self.alpha)

        #join results
        self.df_new_forecast = pd.merge(self.new_yhat, self.new_conf_intervals, how='outer', left_index=True, right_index=True)
        self.df_new_forecast = self.df_new_forecast.rename(columns={'predicted_mean':'New Forecast - Mean','lower sum_all':'New Forecast - Low', 'upper sum_all':'New Forecast - High'})

        #print
        if verbose:
            print(f'\n new forecast: \n {self.df_new_forecast.head(2)}') 
            print(f'\nModel Fit Summary: \n {self.model.summary()}')        

    def join_predictions_with_actuals(self, verbose=False):
        self.df_combined = pd.merge(self.dfreduced, self.df_validation_forecast, how='outer', left_index=True, right_index=True)
        self.df_combined = pd.merge(self.df_combined, self.df_new_forecast, how='outer', left_index=True, right_index=True)
        
        if verbose:
            print(f'\n combined df \n {self.df_combined.head(2)}')

    def run_pipeline(self, verbose=False):
        self.obtain_validation_forecast(verbose=verbose)
        self.obtain_new_forecast(verbose=verbose)
        self.join_predictions_with_actuals(verbose=verbose)


class SarimaxForecast:
    
    '''
    Example Call
    sarimax_forecast = SarimaxForecast(dfreduced=dfreduced, dftrain=dftrain, dftest=dftest, confidence_level=0.95, new_prediction_count=15,order=(2,0,2), seasonal_order=(2,0,2,12))
    sarimax_forecast.run_pipeline(verbose=True)
    '''

    def __init__(self, dfreduced, dftrain, dftest, confidence_level: float, new_prediction_count: int, order: tuple, seasonal_order: tuple):
        self.dfreduced = dfreduced
        self.dftrain = dftrain
        self.dftest = dftest
        self.confidence_level = confidence_level #usually .95
        self.alpha = round(1 - confidence_level,2)
        self.new_prediction_count = new_prediction_count
        self.order = order
        self.seasonal_order = seasonal_order
    
    def obtain_validation_forecast(self, verbose=False):

        #setup model
        model = SARIMAX(
            endog=self.dftrain,
            order=self.order,
            seasonal_order=self.seasonal_order,
            #trend='c',
            dynamic=False
        )
        self.model_fit_validation = model.fit()
        
        #obtain length of validation
        start_forecast_on = min(self.dftest.index)
        end_forecast_on = max(self.dftest.index)

        #obtain forecast, yhat, conf intervals
        forecast = self.model_fit_validation.get_prediction(start_forecast_on, end_forecast_on)
        self.validation_yhat = forecast.predicted_mean
        self.validation_conf_intervals = forecast.conf_int(alpha=self.alpha)

        #join results
        self.df_validation_forecast = pd.merge(self.validation_yhat, self.validation_conf_intervals, how='outer', left_index=True, right_index=True)
        self.df_validation_forecast = self.df_validation_forecast.rename(columns={'predicted_mean':'Validation Forecast - Mean', 'lower sum_all':'Validation Forecast - Low', 'upper sum_all':'Validation Forecast - High'})
        
        #obtain AIC
        aic = self.model_fit_validation.info_criteria('aic')
        print(f'aic: {aic}')

        if verbose:
            print(self.df_validation_forecast.head())

    def obtain_new_forecast(self, verbose=False):

        #setup model
        model = SARIMAX(
            endog=self.dfreduced,
            order=self.order,
            seasonal_order=self.seasonal_order,
            #trend='c',
            dynamic=False
        )
        self.model_fit_new = model.fit()

        #obtain length of validation
        start_forecast_on = max(self.dftest.index)
        end_forecast_on = start_forecast_on + pd.tseries.offsets.MonthEnd(self.new_prediction_count)
        #print(f'start forecast on: {start_forecast_on} \n end forecast on: {end_forecast_on}')

        #obtain forecast, yhat, conf intervals
        forecast = self.model_fit_new.get_prediction(start_forecast_on, end_forecast_on)
        self.new_yhat = forecast.predicted_mean
        self.new_conf_intervals = forecast.conf_int(alpha=self.alpha)

        #join results
        self.df_new_forecast = pd.merge(self.new_yhat, self.new_conf_intervals, how='outer', left_index=True, right_index=True)
        self.df_new_forecast = self.df_new_forecast.rename(columns={'predicted_mean':'New Forecast - Mean','lower sum_all':'New Forecast - Low', 'upper sum_all':'New Forecast - High'})

        #print
        if verbose:
            print(f'\n new forecast: \n {self.df_new_forecast.head(2)}')      

    def join_predictions_with_actuals(self, verbose=False):
        self.df_combined = pd.merge(self.dfreduced, self.df_validation_forecast, how='outer', left_index=True, right_index=True)
        self.df_combined = pd.merge(self.df_combined, self.df_new_forecast, how='outer', left_index=True, right_index=True)
        
        if verbose:
            print(f'\n combined df \n {self.df_combined.head(2)}')

    def run_pipeline(self, verbose=False):
        self.obtain_validation_forecast(verbose=verbose)
        self.obtain_new_forecast(verbose=verbose)
        self.join_predictions_with_actuals(verbose=verbose)

class PlotForecast:
    def __init__(self, forecasting_model, df_newandvalidation_forecasts, y: str):
        self.model = forecasting_model
        self.df_combined = df_newandvalidation_forecasts
        self.y = y

    def residuals(self):
        
        print('\nPlot Residuals')

        #notes
        note1 = 'Looking for constant mean and variance'
        note2 = 'Mean on left graph should be near 0'
        note3 = 'Variance should be uniform'
        print(note1, note2, note3, sep='\n')

        #plot
        plt.rcParams.update({'figure.figsize':(14,3), 'figure.dpi':120})
        dfresiduals = pd.DataFrame(self.model.resid)
        fig, ax = plt.subplots(1, 2)
        dfresiduals.plot(title='Residuals', ax=ax[0])
        dfresiduals.plot(kind='kde', title='Density', ax=ax[1])
        plt.show()
    
    def actuals_validation_and_prediction(self): 
        print('\nPrediction')
        
        fig, ax = plt.subplots(figsize=(15, 5))
        chart = sns.lineplot(x=self.df_combined.index, y=self.y, data=self.df_combined)
        self.df_combined['Validation Forecast - Mean'].plot(ax=ax, color='red', marker='o', legend=True) 
        self.df_combined['New Forecast - Mean'].plot(ax=ax, color='blue', marker=  'o', legend=True)

        chart.set_title('Predictions')
        plt.show()












