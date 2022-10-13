import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit, ShuffleSplit, permutation_test_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score


def split_data(X,Y,group,procedure):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: predictive variable
    Y: predicted variable
    group: group labels used for splitting the dataset
    procedure: strategy to split the data

    Returns
    ----------
    X_train: train set containing the predictive variable
    X_test: test set containing the predictive variable
    y_train: train set containing the predicted variable
    y_test: test set containing the predicted variable
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_idx, test_idx in procedure.split(X, Y, group):
        X_train.append(X[train_idx])
        X_test.append(X[test_idx])
        y_train.append(Y[train_idx])
        y_test.append(Y[test_idx])
    
    return X_train, X_test, y_train, y_test


def verbose(splits, X_train, X_test, y_train, y_test, X_verbose = True, y_verbose = True):
    """
    Print the mean and the standard deviation of the train and test sets
   
    Parameters
    ----------
    splits: number of splits used for the cross-validation
    X_train: train set containing the predictive variable
    X_test: test set containing the predictive variable
    y_train: train set containing the predicted variable
    y_test: test set containing the predicted variable
    X_verbose (boolean): if X_verbose == True, print the descriptive stats for the X (train and test)
    y_verbose (boolean): if y_verbose == True, print the descriptive stats for the y (train and test)
    """
    for i in range(splits):
        if X_verbose:
            print(i,'X_Train: \n   Mean +/- std = ', X_train[i][:][:].mean(),'+/-', X_train[i][:][:].std())
            print(i,'X_Test: \n   Mean +/- std = ', X_test[i][:][:].mean(),'+/-', X_test[i][:][:].std())
        if y_verbose:
            print(i,'y_Train: \n   Mean +/- std = ', y_train[i][:].mean(),'+/-', y_train[i][:].std(), '\n   Skew = ', stats.skew(y_train[i][:]), '\n   Kurt = ', stats.kurtosis(y_train[i][:]))
            print(i,'y_Test: \n   Mean +/- std = ', y_test[i][:].mean(),'+/-', y_test[i][:].std(), '\n   Skew = ', stats.skew(y_test[i][:]), '\n   Kurt = ', stats.kurtosis(y_test[i][:]))
        print('\n')


def compute_metrics(y_test, y_pred, df, fold, print_verbose): 
    """
    Compute different metrics and print them

    Parameters
    ----------
    y_test: ground truth
    y_pred: predicted values
    df: dataFrame containing the result of the metrics
    fold: cross-validation fold for which the metrics are computed
    
    Returns
    ----------
    df_metrics: dataFrame containing the different metrics
    """  

    accuracy = accuracy_score(list(y_test), list(y_pred))
    precision = precision_score(list(y_test), list(y_pred),average = 'macro')
    #roc_auc_ovr = roc_auc_score(list(y_test), list(round_y_pred), multi_class = 'ovo')
    #roc_auc_ovo = roc_auc_score(list(y_test), list(round_y_pred), multi_class = 'ovo')
    print(accuracy,precision)
    print(df)


    df.loc[fold] = [accuracy, precision]
    print(df, 'df')
    if print_verbose:
        print('------Metrics for fold {}------'.format(fold))
        print('accuracy value = {}'.format(accuracy))
        print('precision value = {}'.format(precision))
        #print('roc_auc_ovr value = {}'.format(roc_auc_ovr))
        #print('roc_auc_ovo value = {}'.format(roc_auc_ovo))
        print('\n')

    return df


def reg_PCA(n_component, reg = Lasso()):
    """
    Parameters
    ----------
    n_component: number of components to keep in the PCA

    Returns
    ----------
    pipe: pipeline to apply PCA and Lasso regression sequentially
    """
    estimators = [('reduce_dim', PCA(n_component)), ('clf', reg)] 
    pipe = Pipeline(estimators)
    return pipe

def train_test_classify(X_train, Y_train, gr, C=1.0,splits=5,test_size=0.3, n_components=0.80, random_seed=42, print_verbose=True):
    """
    Build and evaluate a classification model
    
    Parameters
    ----------
    X: predictive variable
    y: predicted variable (binary variable)
    gr: grouping variable
    C: regularization parameter

    Returns
    ----------
    model: list containing the classifier model for each fold
    accuracy: list containing the classifier accuracy across the folds

    See also scikit-learn SVC documentation
    """
    #Initialize the variables
    y_pred = []
    model = []
    model_voxel = []
    accuracy = []
    df_metrics = pd.DataFrame(columns=["accuracy", "precision"])
    splits=5
    test_size=0.3
    n_components=0.80

    #Strategy to split the data
    shuffle_method = GroupShuffleSplit(n_splits = 5, test_size = test_size, random_state = random_seed)
    #X_train, X_test, y_train, y_test = split_data(X, y, gr, shuffle_method)
    x_train, x_test, y_train, y_test = split_data(X_train, Y_train, gr, shuffle_method)

    for i in range(len(x_train)):
        print(x_train[i].shape)
    for i in range(len(y_train)):
        print(y_train[i].shape)
    for i in range(len(x_test)):
        print(x_test[i].shape)
    for i in range(len(y_test)):
        print(y_test[i].shape)


    for i in range(splits):
        ###Build and test the model###
        print("----------------------------")
        print("Training model")
        model_clf = SVC(C=C, kernel="linear",class_weight='balanced', probability = True)
        print(x_train[i].shape, y_train[i].shape)
        model.append(model_clf.fit(x_train[i], list(y_train[i])))

        ###Scores###

        y_pred.append(model[i].predict(x_test[i]))

        #accuracy.append(accuracy_score(list(y_test[i]), list(y_pred[i])))
        fold_df_metrics = compute_metrics(y_test[i], y_pred[i], df_metrics, i, print_verbose)
        df_metrics.loc[i] = [fold_df_metrics['accuracy'],fold_df_metrics['precision']]
        #model_voxel.append(model[i].inverse_transform(model[i].coef_))

    df_metrics.to_csv("dataframe_metrics.csv")
    return x_train, y_train, x_test, y_test, y_pred, model, df_metrics


def compute_permutation(X, y, gr, reg, n_components=0.80, n_permutations=5000, scoring="r2", random_seed=42):
    """
    Compute the permutation test for a specified metric (r2 by default)
    Apply the PCA after the splitting procedure

    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    gr: grouping variable
    n_components: number of components to keep for the PCA
    n_permutations: number of permuted iteration
    scoring: scoring strategy
    random_seed: controls the randomness

    Returns
    ----------
    score (float): true score
    perm_scores (ndarray): scores for each permuted samples
    pvalue (float): probability that the true score can be obtained by chance

    See also scikit-learn permutation_test_score documentation
    """
    if gr == None:
        cv = ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = random_seed)
    else:    
        cv = GroupShuffleSplit(n_splits = 5, test_size = 0.3, random_state = random_seed)
    
    score, perm_scores, pvalue = permutation_test_score(estimator=reg_PCA(n_components,reg=reg), X=X, y=y, groups= gr, scoring=scoring, cv=cv, n_permutations=n_permutations, random_state=42)
    
    return score, perm_scores, pvalue


def predict_on_test(X_train, y_train, X_test, y_test, reg):
    """
    Test the generability of a regression model on left out data
    
    Parameters
    ----------
    X_train: predictive variable to fit the model
    y_train: predicted variable to fit the model
    X_test: predictive variable to predict the model
    y_test: predicted variable to predict the model
    reg: regression technique to perform
    
    Returns
    ----------
    r2: metric to evaluate the performance of the model
    """

    final_model = reg.fit(X_train, y_train)
    r2 = r2_score(y_test, final_model.predict(X_test))
    
    return r2

def bootstrap_test(X,y,gr,reg,splits=5,test_size=0.30,n_components=0.80,n_resampling=1000,njobs=5):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: predictive variable (array-like)
    Y: predicted variable (array-like)
    gr: group labels used for splitting the dataset (array-like)
    reg: regression strategy to use 
    splits: number of split for the cross-validation 
    test_size: percentage of the data in the test set
    n_components: number of components (or percentage) to include in the PCA 
    n_resampling: number of resampling subsets
    njobs: number of jobs to run in parallel

    Returns
    ----------
    bootarray: 2D array containing regression coefficients at voxel level for each resampling (array-like)
    """

    procedure = GroupShuffleSplit(n_splits=splits,test_size=test_size)

    bootstrap_coef = Parallel(n_jobs=njobs,verbose=1)(
        delayed(_bootstrap_test)(
            X=X,
            y=y,
            gr=gr,
            reg=reg,
            procedure=procedure,
            n_components=n_components,
        )
        for _ in range(n_resampling)
    )
    bootstrap_coef=np.stack(bootstrap_coef)

    bootarray = bootstrap_coef.reshape(-1, bootstrap_coef.shape[-1])
    
    return bootarray

def _bootstrap_test(X,y,gr,reg,procedure,n_components):
    """
    Split the data according to the group parameters
    to ensure that the train and test sets are completely independent

    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    gr: group labels used for splitting the dataset
    reg: regression strategy to use
    procedure: strategy to split the data
    n_components: number of components (or percentage) to include in the PCA

    Returns
    ----------
    coefs_voxel: regression coefficients for each voxel
    """
    coefs_voxel = []
    #Random sample
    idx = list(range(0,len(y)))
    random_idx = np.random.choice(idx,
                                  size=len(idx),
                                  replace=True)
    X_sample = X[random_idx]
    y_sample = y[random_idx]
    gr_sample = gr[random_idx]
    
    #Train the model and save the regression coefficients
    for train_idx, test_idx in procedure.split(X_sample, y_sample, gr_sample):
        X_train, y_train = X_sample[train_idx], y_sample[train_idx]
        #X_test, y_test = X_sample[test_idx], y_sample[test_idx]
        model = reg_PCA(n_components,reg=reg)
        model.fit(X_train, y_train)
        coefs_voxel.append(model[0].inverse_transform(model[1].coef_))
        
    return coefs_voxel
