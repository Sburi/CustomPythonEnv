import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class Formats:
    '''
    Purpose
    ----------
        Enable easier formatting to currency, percents, etc for individual numbers.

    Parameters
    ----------
        No class inputs.

    '''
    
    def currency(self, number: float):
        '''
        Purpose
        ----------
            Formats an input number as US currency with 0 decimals.

        Parameters
        ----------
            number: float
                The number you want to convert into currency

        Outputs
        ----------
            A integer formatted as US currency.
        
        '''
        
        try:
            number_rounded = int(round(number))
            number_as_currrency = '${:,.0f}'.format(number_rounded)
            #print(number, '>>', number_as_currrency)
            return number_as_currrency
        except ValueError:
            return number

    def percents(self, number: float, n_decimals: int):
        '''
        Purpose
        ----------
            Formats an input number as a percent with n decimals.

        Parameters
        ----------
            number: float
                The number you want to convert into a percent
            n_decimals: int
                The number of decimals you want the percentage to have.

        Outputs
        ----------
            A integer formatted as a percent with n decimals.
        
        '''
        
        #number = "%.0f%%" % (100 * number) trying to implement number of decimals as below

        number = f"%.{n_decimals}f%%" % (100 * number)
        return number