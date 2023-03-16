import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class Formats:
    '''
    Purpose
    ----------
        Enable easier formatting of individual numbers to currency, percents, etc.

    Parameters
    ----------
        numbers_to_convert: list
            A list with the numbers you want to convert.

    '''
    
    def __init__(self, numbers_to_convert: list):
        self.numbers_to_convert = numbers_to_convert

    def currency(self):
        '''
        Purpose
        ----------
            Formats an input number as US currency with 0 decimals.

        Parameters
        ----------
            None

        Outputs
        ----------
            A list of converted numbers (if conversion possible in every case) or converted/non-converted numbers if unable to convert some/all numbers. 

        Notes
        ----------
            To convert a column to a nicer format, you may use the following:
                pd.options.display.float_format = '{:,.0f}'.format
            The following conversions have been tried to convert columns to a nicer format but did not work
                1
                df[col] = df[col].astype('str').apply(format(','))

                2
                df[col] = df[col].astype('str').str.format(',')

                3
                import locale
                locale.setlocale( locale.LC_ALL, 'English_United States.1252')
                locale._override_localeconv = {'n_sign_posn':1}
                df[col] = df[col].map(locale.currency)

                4
                df[col] = fmt.currency(self.dfcombined[col])
        
        '''
        
        converted_numbers = []
        for number in self.numbers_to_convert:
        
            try:
                number_rounded = int(round(number))
                number_as_currrency = '${:,.0f}'.format(number_rounded)
                converted_numbers.append(number_as_currrency)
            except (ValueError, TypeError):
                converted_numbers.append(number)

        return converted_numbers

    def percents(self, n_decimals: int):
        '''
        Purpose
        ----------
            Formats an input number as a percent with n decimals.

        Parameters
        ----------
            n_decimals: int
                The number of decimals you want the percentage to have.

        Outputs
        ----------
            A list of converted numbers (if conversion possible in every case) or converted/non-converted numbers if unable to convert some/all numbers.
        
        '''
        
        converted_numbers = []
        for number in self.numbers_to_convert:
            number = f"%.{n_decimals}f%%" % (100 * number)
            converted_numbers.append(number)

        return converted_numbers

if __name__ == '__main__':
    def test_currency_conversion():
        numbers = [1, 2, '3a']

        result = Formats(numbers_to_convert=numbers).currency()

        print(result)

    test_currency_conversion()

