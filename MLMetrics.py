#data handlers
import numpy as np
import pandas as pd

#visualizers
import matplotlib.pyplot as plt
import seaborn as sns

#custom
from CustomEnv.Formats import Formats

#error metrics
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error

class ClassificationMetrics:
    '''
    Contains metric methods for evaluating classification models
    Currently only set to work with binary classifiers
    '''
    
    def __init__(self, y_test, y_pred, y_pred_proba, classification_names: list):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.class0name = classification_names[0]
        self.class1name = classification_names[1]

    def obtain_confusion_matrix(self):
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        self.correct_0_predicted_0 = self.cm[0,0]
        self.correct_0_predicted_1 = self.cm[0,1]
        self.correct_1_predicted_0 = self.cm[1,0]
        self.correct_1_predicted_1 = self.cm[1,1]
        print(f'Confusion Matrix: ')# \n {self.cm}')

    def binary_confusion_matrix_describer(self):
        print(f'CORRECT: {self.correct_0_predicted_0} times the class was "{self.class0name}", model predicted "{self.class0name}"')
        print(f'INCORRECT: {self.correct_0_predicted_1} times the class was "{self.class0name}", model predicted "{self.class1name}"')
        print(f'INCORRECT: {self.correct_1_predicted_0} times the class was "{self.class1name}", model predicted "{self.class0name}"')
        print(f'CORRECT: {self.correct_1_predicted_1} times the class was "{self.class1name}", model predicted "{self.class1name}"')
        print()

    def binary_confusion_matrix_visualizer(self):
        df_reframed = pd.DataFrame({
            'Correct Category': [self.class0name, self.class0name, self.class1name, self.class1name],
            'Prediction': ['True', 'False', 'False', 'True'],
            'Value': [self.correct_0_predicted_0, self.correct_0_predicted_1, self.correct_1_predicted_0, self.correct_1_predicted_1],
        })

        sns.catplot(data=df_reframed, x='Correct Category', y='Value', hue='Prediction', kind='bar')
        plt.title('Prediction Accuracy Across Classes')
        plt.show()

    def obtain_classification_report(self):
        cr = classification_report(self.y_test, self.y_pred)
        print(f'Classification Report: \n {cr}')

    def convert_target_to_numbers(self, dict_to_convert_target: dict):
        for correctval, incorrect_val in dict_to_convert_target.items():
            self.y_test = np.where(self.y_test==incorrect_val, correctval, self.y_test)
            self.y_pred = np.where(self.y_pred==incorrect_val, correctval, self.y_pred)

    def ensure_target_is_integer(self):
        self.y_test = self.y_test.astype(int)

    def obtain_roc_auc_score(self):
        #baseline roc auc scores
        dominant_category = int(stats.mode(self.y_test)[0])
        self.y_pred_baseline = [dominant_category for _ in range(len(self.y_test))]
        self.baseline_roc_auc = roc_auc_score(self.y_test, self.y_pred_baseline)
        self.baseline_fpr, self.baseline_tpr, _ = roc_curve(self.y_test, self.y_pred_baseline)
        
        #roc auc score
        self.actual_roc_auc = round(roc_auc_score(self.y_test, self.y_pred_proba),2)
        self.actual_fpr, self.actual_tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        
        print('ROC AUC Score:')

        #plot
        plt.plot(self.baseline_fpr, self.baseline_tpr, linestyle='--', label='baseline')
        plt.plot(self.actual_fpr, self.actual_tpr, marker='.', label='Actual')
        plt.title('ROC AUC Score')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

        #show scores
        print(f'Baseline ROC AUC: {self.baseline_roc_auc}')
        print(f'ROC AUC Score: {self.actual_roc_auc}')


class ForecastingMetrics:

    '''
    fmetrics = ForecastingMetrics(dftest=sarimax_forecast.dftest, validation_yhat=sarimax_forecast.validation_yhat, model_fit_validation=sarimax_forecast.model_fit_validation)
    fmetrics.run_pipeline_forecasting_error_metrics(type='money')
    '''

    def __init__(self, dftest, validation_yhat, model_fit_validation):
        self.dftest=dftest
        self.validation_yhat=validation_yhat
        self.model_fit_validation = model_fit_validation

    def show_summary(self):
        return print(self.model_fit_validation.summary())

    def calc_mean(self, type=''):

        #obtain mean, as reference point for MSE
        mean = self.dftest.values.mean()
        mean = round(mean, 2)
        self.mean = mean

        #apply formatting
        if type=='money':
            frmt = Formats()
            mean = frmt.currency(mean)

        print(f'Mean: {mean}')
        
    def calc_mean_squared_error(self, type=''):

        #obtain MSE
        mse = mean_squared_error(self.dftest.values, self.validation_yhat, squared=True)
        mse = round(mse, 2)

        #apply formatting
        if type=='money':
            frmt = Formats()
            mse = frmt.currency(mse)

        #print
        print(f'Mean Squared Error:  {mse}')

    def calc_root_mean_squared_error(self, type=''):
        
        #obtain MSE
        rmse = mean_squared_error(self.dftest.values, self.validation_yhat, squared=False)
        rmse = round(rmse, 2)
        self.rmse = rmse

        #apply formatting
        if type=='money':
            frmt = Formats()
            rmse = frmt.currency(rmse)

        #print
        print(f'Root Mean Squared Error:  {rmse}')

    def mean_percent_of_rmse(self):

        percent = self.rmse / self.mean

        frmt = Formats()
        percent = frmt.percents(number=percent, n_decimals=2)

        print(f'Mean % of RMSE: {percent}')

    def run_pipeline_forecasting_error_metrics(self, type=''):
        
        #self.calc_mean_squared_error(type=type)
        self.calc_root_mean_squared_error(type=type)
        self.calc_mean(type=type)
        self.mean_percent_of_rmse()
        self.show_summary()




















