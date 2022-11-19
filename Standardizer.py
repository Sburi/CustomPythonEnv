
class Standardize:

    def __init__(self, df):
        self.df = df

    def standardize_column_values(self, conversion_dict: dict, current_col: str, revised_col='', print_conversions=False):
        '''
        Purpose \n
        Provides system that loops through a dictionary and provided column inputs. If dictionary value is found, value is converted to dictionary key.

        Example \n
        Given the below dictionary, this method will loop through the current_col looking for the dictionary values 'usrname' and 'uname', if it finds those values, it will convert them to 'Username' in the revised_col. Conversions may also be printed for reference.
        dict = {
            'Username' : ['usrname', 'uname'],
        }

        Output \n
        Modifies this classes self.df. The sub-calls of this standardizer directly return df for their specific use-cases. If this method is used independently and fed a dictionary, then you can extract the dataframe by obtaining the .df attribute of this class.

        '''
    
        #if no revised_col_name has been provided, overwrite the original vendor name with the revised vendor name column
        if revised_col == '':
            revised_col = current_col

        #normalize vendors
        temp_conversion_col = 'Converted To'
        for k,v in conversion_dict.items():
            self.df.loc[
                self.df[current_col].str.lower().isin([i.lower() for i in v]),
            temp_conversion_col] = k
        self.df[temp_conversion_col] = self.df[temp_conversion_col].fillna(self.df[current_col])

        #print changes
        if print_conversions:
            conversions =  self.df.where(self.df[current_col] != self.df[temp_conversion_col])[[current_col, temp_conversion_col]]#
            conversions.rename(columns={current_col:'Original Entry'}, inplace=True)
            print('conversions:')
            print(conversions.drop_duplicates().dropna().sort_values('Converted To'))

        #implement change
        self.df[revised_col] = self.df[temp_conversion_col]
        self.df = self.df.drop(temp_conversion_col, axis=1)

    def vendors(self, current_col, revised_col, print_conversions=False):
        '''
        Purpose \n
        Standardizes vendors.
        
        Inputs \n
        :current_col: the column with the values you want to standardize
        :revised_col: the column where you want the new, standardized values to appear. May be the same as the current_col if you're ok with overwriting those values.
        :print_conversions: Boolean toggling whether or not from-->to conversions are shown.

        Returns
        dataframe
        '''
        conversion_dictionary = {
            'ActiveOps': ['Activeops USA', 'Activeops Usa Inc.'],
            'Aspect': ['Aspect Software', 'Aspect Software Inc'],
            'Avaya': ['Avaya Inc'],
            'Axim': ['Axim Inc'],
            'Citrix': ['Citrix Systems Inc'],
            'Clarabridge': ['Clarabridge Inc'],
            'Cogito': ['Cogito Corporation', 'Cogito Corp'],
            'Concentrix': ['Concentrix Cvg Llc Fka Intervoice Llc'],
            'Convergent': ['Convergent Solutions Group Llc Dba Csg Global Consulting'],
            'Eliassen': ['Eliassen Group, Llc'],
            'Enclara': ['Enclara Budget'],
            'Enghouse': ['Enghouse Interactive', 'Enghouse Interactive Inc.'],
            'Five9': ['Five9 Inc'],
            'Genesys': ['GENESYS CLOUD SERVICES INC', 'Genesys Telecommunications Laboratories Inc', 'Genesys Telecommnunications Laboratories, Inc'],
            'IBM': ['Ibm Corporation'],
            'InMoment': ['Inmoment, Inc'],
            'Intradiem': ['Intradiem, Inc', 'Intradiem Inc'],
            'Mattersight': ['Mattersight Corporation', ''],
            'MPulse': ['Mpulse Mobile Inc'],
            'Nuance': ['NUANCE COMMUNICATIONS, INC.', 'Nuance Communications'],
            'Oracle': ['Oracle America Inc'],
            'Qualtrics': ['Qualtrics, Llc'],
            'Salesforce': ['Salesforce.Com, Inc.'],
            'Syncsort': ['Syncsort Incorporated', 'Syncsort Incorporated Nj'],
            'TCS': ['TATA CONSULTANCY SERVICES LIMITED', 'TATA CONSULTING SERVICES LIMITED', 'Tcs '],
            'Twilio': ['Twilio Inc'],
            'Verint': ['Verint Americs Inc', 'Verint Americas Inc'],
            'Vs Brooks Inc': ['Vs Brooks Inc'],
            'World Wide Technologies': ['Wwt'],
        }

        self.standardize_column_values(conversion_dict=conversion_dictionary, current_col=current_col, revised_col=revised_col, print_conversions=print_conversions)
        return self.df

    def transaction_classes(self, current_col, revised_col, print_conversions=False):
        '''
        Purpose \n
        Standardizes transaction classes.
        
        Inputs \n
        :current_col: the column with the values you want to standardize
        :revised_col: the column where you want the new, standardized values to appear. May be the same as the current_col if you're ok with overwriting those values.
        :print_conversions: Boolean toggling whether or not from-->to conversions are shown.

        Returns
        dataframe
        '''

        conversion_dictionary = {
            'Hardware': ['Hardware - Server/OS'],
            'Software': ['Software - Software as a Service', 'Software - Perpetual'],
            'Professional Services': ['Professional Services'],
            'Direct Labor': ['Direct Labor', 'Software Development (Labor)'],
        }

        self.standardize_column_values(conversion_dict=conversion_dictionary, current_col=current_col, revised_col=revised_col, print_conversions=print_conversions)
        return self.df

    def prioritization_categories(self, current_col, revised_col, print_conversions=False):
        '''
        Purpose \n
        Standardizes prioritization categories.
        
        Inputs \n
        :current_col: the column with the values you want to standardize
        :revised_col: the column where you want the new, standardized values to appear. May be the same as the current_col if you're ok with overwriting those values.
        :print_conversions: Boolean toggling whether or not from-->to conversions are shown.

        Returns
        dataframe
        '''

        conversion_dictionary = {
            'Objective': ['Objective'],
            'ABO': ['ABO'],
            'IOP': ['IOP'],
            'RUN': ['RUN'],
        }

        self.standardize_column_values(conversion_dict=conversion_dictionary, current_col=current_col, revised_col=revised_col, print_conversions=print_conversions)
        return self.df   

    def funding_portfolio(self, current_col, revised_col, print_conversions=False):
        '''
        Purpose \n
        Standardizes funding portfolio.
        
        Inputs \n
        :current_col: the column with the values you want to standardize
        :revised_col: the column where you want the new, standardized values to appear. May be the same as the current_col if you're ok with overwriting those values.
        :print_conversions: Boolean toggling whether or not from-->to conversions are shown.

        Returns
        dataframe
        '''

        conversion_dictionary = {
            'AEP Readiness': ['Run AEP Readiness'],
            'CECP': ['Customer Experience Platform'],
            'CRM': ['CRM (Group)'],
            'Desktop Experience': ['Desktop Experience'],
            'Digital Service': ['Digital Service'],
            'Enable Communication Experience': ['Enable Communication Experience', 'Enable Communications Platform (DIHE)'],
            'Enterprise Feedback Loop': ['Enterprise Feedback Loop'],
            'Humana in your neighborhood (DIHE)': ['Humana in your neighborhood (DIHE)'],
            'Idle Time Push Learning ABO': ['?'],
            'Machine Learning Platform (MLP)': ['Machine Learning Platform (MLP)'],
            'Marketing': ['?'],
            'Marketing (Run)': ['?'],
            'Marketing RUN Compliance': ['Marketing RUN Compliance'],
            'NLP, Deep Learning, Voice Analytics': ['NLP, Deep Learning, Voice Analytics'],
            'Pharmacy - Humana Pharmacy': ['Pharmacy - Humana Pharmacy'],
            'Pharmacy Simplified Consumer Experience': ['Pharmacy Simplified Consumer Experience'],
            'Service Technologies': ['Service Technologies'],
            'Voice and WFM': ['Voice and Workforce Management'],
            'Voice Platform': ['Voice Platform'],
            'Workforce Management (Grow DIHE)': ['?'],
        }

        self.standardize_column_values(conversion_dict=conversion_dictionary, current_col=current_col, revised_col=revised_col, print_conversions=print_conversions)
        return self.df   

    def template(self, current_col, revised_col, print_conversions=False):
        '''
        Purpose \n
        Standardizes ___.
        
        Inputs \n
        :current_col: the column with the values you want to standardize
        :revised_col: the column where you want the new, standardized values to appear. May be the same as the current_col if you're ok with overwriting those values.
        :print_conversions: Boolean toggling whether or not from-->to conversions are shown.

        Returns
        dataframe
        '''

        conversion_dictionary = {
            '': [''],
            '': [''],
            '': [''],
            '': [''],
        }

        self.standardize_column_values(conversion_dict=conversion_dictionary, current_col=current_col, revised_col=revised_col, print_conversions=print_conversions)
        return self.df   

if __name__ == '__main__':

    def test_simplify_vendors():
        dfTest = pd.DataFrame()
        dfTest['Vendor Name'] = ['Wwt', 'Verint', 'Nuance Communicatio', 'Nuance Communications, Inc.', 'Intradiem Inc', 'Intradiem Inc', 'Enclara Budget', 'Mattersight Corporation']
        #test = test_simplify_vendors(dfTest, vendor_col='Vendor Name', print_conversions=True)