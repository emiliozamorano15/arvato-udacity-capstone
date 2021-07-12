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

def model_performance(X, y, model):
    """
    Run a series of cross_validation/training splits and plot the 
    resulting AUC score
    
    INPUT:
        X: Predictors Matrix
        y: Target Vetor
        estimator: sklearn predictor object with fit and predict methods
    """
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    
    num_samples = 5

    train_sizes, train_scores, test_scores = learning_curve(
        model
        , X.iloc[X.sample(frac=1, random_state = 42).index]
        , y.iloc[X.sample(frac=1, random_state = 42).index]
        , cv = 3
        , scoring = 'roc_auc'
        , train_sizes=np.linspace(0.1, 1.0, num_samples)
        , verbose = 4
        , n_jobs= -1
        )

    train_scores_mean = np.mean(train_scores, axis=1)
    cv_scores_mean = np.mean(test_scores, axis=1)
    print("AUC train = {}".format(train_scores_mean[-1].round(2)))
    print("AUC cv = {}".format(cv_scores_mean[-1].round(2)))


    plt.plot(np.linspace(0.1, 1.0, num_samples)*100, train_scores_mean,
             label="Training score")
    plt.plot(np.linspace(0.1, 1.0, num_samples)*100, cv_scores_mean,
             label="Cross-validation score")

    plt.title("Learning Curves")
    plt.xlabel("% of training set")
    plt.ylabel("AUC")

    plt.legend(loc="best")
    print("")
    plt.show()

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        verbose=0):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Source: [https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html]
    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

if __name__ == '__main__':
    pass
