import pandas as pd
import numpy as np

def find_cat_cols(df):
    '''
    Function to find the names of categorical columns
    
    INPUT
    df: pandas dataframe

    OUTPUT
    cat_cols: list of names of columns with categorical values
    '''

    cat_cols = list(df.select_dtypes(['object']).columns)

    return cat_cols

    
def find_binary_cols(df):
    '''
    Function to find the names numerical columns with binary (1/0) values

    INPUT 
    df: pandas dataframe
    
    OUTPUT
    bin_cols: list of names of columns with binary values
    '''

    bin_cols = []
    for col in df.select_dtypes(['float64', 'int64']).columns:
        n_unique = df[col].dropna().nunique()
        if n_unique == 2:
            bin_cols.append(col)

    return bin_cols
    

def clean_data(df, drop_rows = [], drop_cols = []):
    '''
    Function to clean Arvato's datasets. It mainly changes data format for certain columns,
    and drops columns (rows) which exceed a given threshold of missing values.

    INPUT
    df: pandas dataframe (from Arvato's )
    drop_rows: list of row indices to drop
    drop_cols: list of col names to drop

    OUTPUT
    clean_df: pandas dataframee with cleaned data
    '''

    if len(drop_cols) > 0:
        clean_df = df.drop(drop_cols, axis = 1)
    
    if len(drop_rows) > 0:
        clean_df = clean_df.loc[~clean_df.index.isin(drop_rows)]

    ## Transform EINGEFUEGT_AM to date format (only year part)
    clean_df['EINGEFUEGT_AM'] = pd.to_datetime(clean_df['EINGEFUEGT_AM'], format = '%Y-%m-%d').dt.year
    
    ### Label-encode OST_WEST_KZ
    clean_df['OST_WEST_KZ'] = clean_df['OST_WEST_KZ'].replace('W',1).replace('O', 0)
    clean_df['OST_WEST_KZ'] = pd.to_numeric(clean_df['OST_WEST_KZ'], errors = 'coerce')

    return clean_df

if __name__ == '__main__':
    pass
