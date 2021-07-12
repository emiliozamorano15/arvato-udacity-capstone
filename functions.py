import pandas as pd
import numpy as np

def missing_dict(df):
    '''
    Function to build a dictionary of indicators of missing information per feature

    INPUT:
    df: pandas dataframe with features, description, and values that mean "unknown"
    
    OUPUT:
    missing_dict: dictionary of values for "unkwon" per feature
    '''

    unknown_values = []
    for val in df.Value:
        ## evaluate whether missing 'value' is an integer (one digit)
        if isinstance(val, int):
            unknown_values.append([val])
        ## evaluate whether attribute has more than one value (a string object in the dataframe)
        elif isinstance(val, str):
            split_list = val.split(',')
            int_list = [int(x) for x in split_list]
            unknown_values.append(int_list)

    unknown_dict = {}
    for attr, value_list in zip(df.Attribute, unknown_values):
        unknown_dict[attr] = value_list

    unknown_dict['ALTERSKATEGORIE_FEIN'] = [0]
    unknown_dict['GEBURTSJAHR'] = [0]
    
    return unknown_dict
    

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

    ## Cast CAMEO_DEUG_2015 to int
    clean_df['CAMEO_DEUG_2015'] = clean_df['CAMEO_DEUG_2015'].replace('X',np.nan)
    clean_df['CAMEO_DEUG_2015'] = clean_df['CAMEO_DEUG_2015'].astype('float')

    ## Transform EINGEFUEGT_AM to date format (only year part)
    clean_df['EINGEFUEGT_AM'] = pd.to_datetime(clean_df['EINGEFUEGT_AM'], format = '%Y-%m-%d').dt.year
    
    ### Label-encode OST_WEST_KZ
    clean_df['OST_WEST_KZ'] = clean_df['OST_WEST_KZ'].replace('W',1).replace('O', 0)
    clean_df['OST_WEST_KZ'] = pd.to_numeric(clean_df['OST_WEST_KZ'], errors = 'coerce')

    return clean_df


def scree_plot(pca):
    """
    Function to make a scree plot out of a PCA object
    
    INPUT
    pca: PCA fitted object

    OUTPUT
    scree plot
    """
    import matplotlib.pyplot as plt

    nc = len(pca.explained_variance_ratio_)
    ind = np.arange(nc)
    vals = pca.explained_variance_ratio_
    cumvals = np.cumsum(vals)

    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot()
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)

    plt.xlabel('No. of Components')
    plt.ylabel('Cum. explained variance')
    plt.title('Scree plot PCA')

def get_cluster_centers(cluster_pipeline, num_cols, col_names):
    """
    Function inverse transform pca components. 
    
    INPUT:
        cluster: object of cluster_pipeline
        num_cols: list of numerical attributes which were rescaled
        col_names: names of all columns after Column Transformer operation
        
        
    OUTPUT:
        df (DataFrame): DataFrame of cluster_centers with their attributes values
        
    """

    pca_components = cluster_pipeline.named_steps['reduction']
    kmeans = cluster_pipeline.named_steps['clustering']
    transformer =  cluster_pipeline.named_steps['transform']

    centers = pca_components.inverse_transform(kmeans.cluster_centers_)
    df = pd.DataFrame(centers, columns = col_names)

    num_scale = transformer.named_transformers_['num'].named_steps['num_scale']

    df[num_cols] = num_scale.inverse_transform(df[num_cols])
 
    return df

def model_performance(X, y, model, num_samples):
    """
    Draw learning curve that shows the validation and training auc_score of an estimator 
    for varying numbers of training samples.
    
    INPUT:
        X: Predictors Matrix
        y: Target Vetor
        estimator: sklearn predictor object with fit and predict methods
        num_samples: number of training samples to plot
        
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt

    train_sizes, train_scores, test_scores = learning_curve(
        model
        , X
        , y
        , scoring = 'roc_auc'
        , train_sizes=np.linspace(.1, 1.0, num_samples)
        )

    train_scores_mean = np.mean(train_scores, axis=1)
    cv_scores_mean = np.mean(test_scores, axis=1)
    print("AUC train = {}".format(train_scores_mean[-1].round(2)))
    print("AUC cv = {}".format(cv_scores_mean[-1].round(2)))


    plt.plot(np.linspace(.1, 1.0, num_samples)*100, train_scores_mean,
             label="Training score")
    plt.plot(np.linspace(.1, 1.0, num_samples)*100, cv_scores_mean,
             label="Cross-validation score")

    plt.title("Learning Curves")
    plt.xlabel("% of training set")
    plt.ylabel("AUC")

    plt.legend(loc="best")
    print("")
    plt.show()


if __name__ == '__main__':
    pass
