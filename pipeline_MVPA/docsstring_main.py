"""
    Arguments
    --------
    root_dir : directory to all the subject's fmri volumes. This script is built assuming that root dir is a dir with a folder for each participant.
    timestamps_path : Path to all the timestamps in a folder. The script identify which timestamps to choose based on its name. Can be .mat or .csv
    dir_to_save : Directory to save the statistical contrast maps. It will only be used if contrast_type is not 'None'.
    contrast_type : Type of contrast to compute. Choices=['all_shocks','each_shocks','suggestions']. By default None.
    parser : True by default. If False, parser.args will be overlooked and user will be able to manually provide the necessary arguments while calling main().
    compute_DM : True by default but if False, a path to the design matrices and the concatenated fmri timeseries is expected.

    ---------------


    reg parameter
    -------------
    Algorithm to use on the data

    'lasso': Apply Lasso regression
    'ridge': Apply Rigde regressions
    'svr': Apply a Support Vector Regression
    'svc': Apply a Support Vector Classifier
    'lda': Apply a Linear Discriminant Analysis classifier
    'rf': Apply a Random Forest classifier
    'huber': Apply a Robust Huber Regression
    'linear': Apply a linear regression

    analysis parameter
    ------------------
    Specify which kind of analysis to run on the data between 3 choices:

    'regression': regression analysis
    'classification': classification analysis
    'sl': searchlight analysis

    folder
    ------
    Where to save the data
    """
