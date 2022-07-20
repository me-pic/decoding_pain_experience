
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, mean_img

def create_DM(subject_data, timestamps, DM_name, df_mvmnt_reg):

    """
    Description
    A function that computes a design matrix using the nilearn function make_first_level_design_matrix.
    *Some parameters must be adapted to the study design*
    It returns a pandas design matrix and a 4D time series of the subject nii data.

    Variables

    subject_data : A list with all the paths to nii images
    timestamps_path : Path to timestamps
    DM_name : Name of the design matrix, under which it will be saved
    regresseurs_mvmnt : matrix for the mouvment regressors
    """
    print('====================================')
    print('COMPUTING design matrix under name : ' + DM_name)

    #Extraction of subject'volumes (4D nii file)
    fmri_time_series = concat_imgs(subject_data)

    #////////////TIMESTAMPS////////////////
    if type(timestamps) is str:

        timestamps = pd.read_excel(timestamps_path, header=None)
        #formatage des type des entrées et insertion de titres
        timestamps = pd.DataFrame.transpose(events)
        timestamps.rename(columns = {0:'onset', 1:'duration', 2:'trial_type'}, inplace = True)

    timestamps['onset'] = timestamps['onset'].astype(np.float64)
    timestamps['duration'] = timestamps['duration'].astype(np.float64)
    timestamps['trial_type'] = timestamps['trial_type'].astype(np.str)

    #!!change the tr and function's parameters according to study design!!
    tr = 3.
    n_scans = len(subject_data)
    frame_times = np.arange(n_scans) * tr

    design_matrix = make_first_level_design_matrix(
                frame_times,
                timestamps,
                hrf_model='spm',
                drift_model='cosine',
                high_pass=.00233645,
                add_regs = df_mvmnt_reg) #array of shape(n_frames, n_add_reg)

    #================Prints for info================

    print('SHAPE of TIMESPTAMPS is {0}'.format(timestamps.shape))
    print('SHAPE of MOVEMENT REGRESSORS: {} '.format(df_mvmnt_reg.shape))
    print('SHAPE OF fMRI TIMESERIES : {} '.format(fmri_time_series.shape))
    print('SHAPE OF DESIGN MATRIX : {} '.format(design_matrix.shape))

    #================plot option================
    #from nilearn.plotting import plot_design_matrix
    #plot_design_matrix(design_matrix)
    #import matplotlib.pyplot as plt
    #plt.show()
    #design_matrix.shape


    return design_matrix, fmri_time_series

