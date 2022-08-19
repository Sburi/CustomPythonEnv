class Formats:
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    
    def currency(number):
        try:
            number_rounded = int(round(number))
            number_as_currrency = '${:,.0f}'.format(number_rounded)
            #print(number, '>>', number_as_currrency)
            return number_as_currrency
        except ValueError:
            return number