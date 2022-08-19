import pandas as pd

class ExcelMods:
    def excel_setcolwidth(filename, **dfs):
        '''
        sets column width of excel files
        filename: input fully qualified file location such as r'C:\\location\\file.xlsx'
        dfs: input dataframes to parse over as dictionary, such as {'name_of_df1':df1, 'name_of_df2':df2}
        input dataframes to iterate over as 
        '''
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        for sheetname, df in dfs.items():  # loop through `dict` of dataframes
            df.to_excel(writer, sheet_name=sheetname, index=False)  # send df to writer
            worksheet = writer.sheets[sheetname]  # pull worksheet object
            for idx, col in enumerate(df.columns):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width
        writer.save()
    #example call
    # dfs = {'Dim': dfFinalDim, 'Fact': dfFinalFact}
    # filename=r'C:\Users\SIB4953\Humana\Forecast\Python Processed Datasets\Multiyear Forecast.xlsx'
    # excel_setcolwidth(filename, **dfs)