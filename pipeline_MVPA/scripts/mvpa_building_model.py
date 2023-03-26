import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit, ShuffleSplit, permutation_test_score
from sklearn.metrics import roc_auc_score,roc_curve, accuracy_score, balanced_accuracy_score, top_k_accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import FastICA
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

#----------------------------------------------------------------------------------------------------------------------------
def compute_metrics_y_pred(y_test, y_pred, verbose):
    """
    Compute different metrics and print them

    Parameters
    ----------
    y_test: ground truth
    y_pred: predicted values
    df_metrics: dataFrame containing the metrics names
    fold: cross-validation fold for which the metrics are computed

    Returns
    ----------
    df_metrics: dataFrame containing the different metrics
    """

    accuracy = accuracy_score(list(y_test), list(y_pred))
    balanced_accuracy = balanced_accuracy_score(list(y_test),list(y_pred))
    precision = precision_score(list(y_test), list(y_pred),average = 'macro')
    #precision_recall_curve = precision_recall_curve()

    cm = confusion_matrix(list(y_test), list(y_pred))
    cr = classification_report(list(y_test), list(y_pred))
    if verbose:
        print('--------')
        print('confusion matrix with shape : {} and being type : {}'.format(cm.shape, type(cm)))
        print('Classif report having shape : {} and being type : {}'.format(cr, type(cr)))

    row_metrics = [accuracy, balanced_accuracy, precision]

    return row_metrics, cm, cr


def train_test_models(X_train, Y_train, gr, n_folds, C=1.0,test_size=0.3, n_components_pca = 0.80, random_seed=42, verbose = False, binary = False):
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
    decision_func_df = []
    roc_ovo_df = []
    y_scores = []
    folds_names = ['fold{}'.format(i+1) for i in range(n_folds)]
    final_model_name = ['final_model'] # initialization for the metrics df colnames
    df_y_pred = pd.DataFrame(columns = folds_names)
    models = []
    model_voxel = []
    metrics_index_names = folds_names + final_model_name
    accuracy = []
    balanced_accuracy = []
    precision = []
    ls_roc_auc_ovo = []
    dict_confusion = {'fold{}'.format(i+1): i for i in range(n_folds)}
    dict_classif_report = {'fold{}'.format(i+1): i for i in range(n_folds)}
    dict_decision_func = {'fold{}'.format(i+1): i for i in range(n_folds)}

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision'] #'confusion_matrix', 'classif_rapport']
    df_metrics = pd.DataFrame(index = (metrics_index_names), columns= metrics_colnames)
    #['accuracy', 'balanced_accuracy','top_k_accuracy', 'average_precision', 'f1_score',
    #'f1_micro', 'f1_macro', 'precision_score', 'precision_recall_curve', 'average_precision_score',
    # 'roc_auc_ovo'])

    #Strategy to split the data
    shuffle_method = GroupShuffleSplit(n_splits = n_folds, test_size = test_size, random_state = random_seed)
    x_train, x_test, y_train, y_test = split_data(X_train, Y_train, gr, shuffle_method)

    # Principal component (PC)
    PC_var_ls = []
    PC_values_ls = []
    ica = False
    for i in range(n_folds):

        if n_components_pca[i] <1 and ica == False:
            print('PCA with {} components'.format(n_components_pca[i]))
            pca = PCA(n_components = n_components_pca[i])
            print(x_train[i]) # visu test
            x_train[i] = pca.fit_transform(x_train[i])
            x_test[i] = pca.transform(x_test[i])
            PC_values =  np.arange(pca.n_components_) + 1
            PC_values_ls.append(PC_values)
            PC_var = pca.explained_variance_ratio_
            PC_var_ls.append(PC_var)

        # Independent component analysis
        if ICa == True:
            ICA = FastICA(whiten='unit-variance')
            x_train[i] = ICA.fit_transform(x_train[i])
            x_test[i] = ICA.transform(x_test[i])

        model_clf = SVC(kernel='linear',gamma='auto', probability = True,class_weight=None)
        model_clf.fit(x_train[i], list(y_train[i]))
        models.append(model_clf)

        # Metrics
        # y_pred
        y_pred.append(models[i].predict(x_test[i])) # prediction ith the test set

        # decision_function
        decision_func = model_clf.decision_function(x_test[i])
        decision_func_df.append(decision_func)
        dict_decision_func['fold{}'.format(i+1)] = decision_func

        # Metrics using y_score
     	# ROC
        #y_test = y_test[i].astype(int)
        if binary:
            y_score = model_clf.predict_proba(x_test[i])[:, 1]
            roc_auc_ovo = roc_auc_score(list(y_test[i]), np.array(y_score))
        else:
            y_score = model_clf.predict_proba(x_test[i])
            roc_auc_ovo = roc_auc_score(list(y_test[i]), np.array(y_score), multi_class = 'ovo')

        ls_roc_auc_ovo.append(roc_auc_ovo)

        fold_row_metrics, cm, cr = compute_metrics_y_pred(y_test[i], y_pred[i],verbose = verbose)
        dict_confusion['fold{}'.format(i+1)] = cm
        dict_classif_report['fold{}'.format(i+1)] = cr


        # Saving metrics in dataframe
        df_metrics.loc['fold{}'.format(i+1), metrics_colnames] = fold_row_metrics
        #model_voxel.append(model[i].inverse_transform(model[i].coef_))
        if verbose:
            print("----------------------------")
            print('Training model in fold{}'.format(i+1))
            print(cm)
            print('roc_auc_ovo :  {} and list of roc auc {}'.format(roc_auc_ovo,ls_roc_auc_ovo))
            print('fold_row_metrics :  {} and its shape {}'.format(fold_row_metrics,len(fold_row_metrics)))

    if verbose:
        print('x_train dim : {}'.format([x_train[i].shape for i in range( len(x_train[i])) ]))
        print('x_test dim : {}'.format([x_test[i].shape for i in range( len(x_test[i])) ]))
        print('y_train dim : {}'.format([y_train[i].shape for i in range( len(y_train[i])) ]))
        print('y_test dim  : {}'.format([y_test[i].shape for i in range( len(y_test[i])) ]))


    fold_results = dict(pca_n_components = n_components_pca, PC_values = PC_values_ls,PC_var = PC_var_ls, x_train = x_train, y_train = y_train, x_test = x_test, y_pred = y_pred, fold_models = models, df_fold_metrics = df_metrics,roc_auc_ovo = ls_roc_auc_ovo, confusion_matrix = dict_confusion, classification_report = dict_classif_report, decision_function = decision_func_df)

    #df_metrics.to_csv("dataframe_metrics.csv")
    return fold_results


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


def hot_split_Y_test(Y_test, n_classes):

# function to split Y_test, a row vector into a array where each col contains zeros and a value in the row Y_test.
# E.g. Y_test is shape (83,1), with values of 1,2,3,4, output will have shape (84,4), with fist col looking like : [[1 0 0 1 0 ...], [0,2,2,0,...],...]

    Y_test_wide = np.zeros([len(Y_test),n_classes])

    for cl in range(1,n_classes+1): # assuming 4 classes
        idx = 0
        col = np.zeros(len(Y_test))

        for element in Y_test:

            if element == cl: # adding value to vector to col
                col[idx] = Y_test[idx] # col to stack in Y_test4
            idx += 1

        Y_test_wide[:,cl-1] = col
    return Y_test_wide


def reg_PCA(n_component, clf = SVC()):
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
