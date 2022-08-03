
#main path to data, change according to environment
root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test'
dir_to_save = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM'
timestamps_root_path = r'C:\Users\Dylan\Desktop\BAC_neurocog\UM_H22\PSY3008\times_stamps'
#root_dir = r'/data/rainville/dylan_projet_ivado_decodage/Nii'
#dir_to_save = r'/data/rainville/dylan_projet_ivado_decodage/results/GLM_1st_level_hyperHypo_Neut_all_shocks/'
#timestamps_root_path = r'/data/rainville/dylan_projet_ivado_decodage/time_stamps'

import numpy as np
import os
import pandas as pd
import glob
import nibabel as nib
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
from B_design_matrix import function_DM as B_design_matrix
from C_contrasts import function_contrasts as contrast


def main(root_dir = None, timestamps_path = None, dir_to_save= None, contrast_type = None, parser = True, compute_DM = True):


    #--------Parser--------
    #Argument parser
    if __name__ == "__main__":
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser.add_argument("--root_dir", type=str) #dir to the subjects' files containing the fmri data
        parser.add_argument("--dir_to_save", type=str) #path to save the output
        parser.add_argument("--timestamps_path_root", type=str) #path to the timestamps files
        parser.add_argument("--many_runs", type=str)
        parser.add_argument('--contrast_type', type=str, choices=['all_shocks','each_shocks','suggestions'], default='all_shocks')
        args = parser.parse_args()

    #==============================================================
    #store all subject's name and paths in lists
    ls_subj_name = [subject for subject in os.listdir(root_dir)]
    ls_subj_path  = glob.glob(os.path.join(root_dir,'*'))
    contrast_paths = []
    if contrast_type != None: #make a result directory for the contrasts
        path_contrast_results = os.path.join(dir_to_save,contrast_type) #creating a root dir to save all the contrast
        if os.path.exists(path_contrast_results) is False:
            os.mkdir(path_contrast_results)
        print(path_contrast_results)

    for subj_path in ls_subj_path:
        subj_name = os.path.basename(os.path.normpath(subj_path))
        print('At : ' + subj_name)

        #-----get design matrix and timeseries------
        if compute_DM != True:
            design_matrices = glob.glob(os.path.join(compute_DM, subj_name,'DM*'))
            fmri_time_series = glob.glob(os.path.join(compute_DM, subj_name, '*fmri*'))
            conditions = 'hyper_hypo'
            print(design_matrices)
        else:
            B_design_matrix.check_if_empty(root_dir)
            save_DM = os.path.join(dir_to_save,'DM_timeseries')
            if os.path.exists(save_DM) is False:
                os.mkdir(save_DM)
            design_matrices, fmri_time_series, conditions = B_design_matrix.compute_DM(subj_path,timestamps_path,3, save = save_DM)
        print(design_matrices[0])
        print('HERRRRE')
        print(type(design_matrices[0]))
        print(design_matrices[0].shape)
        #-----Contrast-----
        if contrast_type == 'each_shocks':
            beta_map, contrast_path = contrast.glm_contrast_1event(design_matrices,
             path_contrast_results, subj_name, fmri_time_series, run_name = conditions)
        elif contrast_type == 'suggestions':
            pass#***in developpement***
        elif contrast_type == 'all_shocks': #default = all_shocks
            beta_map, contrast_path = contrast.glm_contrast_runs_all_shocks(design_matrices,
             fmri_time_series, path_contrast_results, subj_name, run_name = conditions)
        else:
            print('skipping contrast')

        if contrast_type != None : #keep track of the contrasts' paths to save them
            contrast_paths.append(contrast_path)
        print('contrast_paths lenght : ', len(contrast_paths))

    #==============================================================
    #Second level analysis

compute_DM = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\DM_timeseries'
#contrast_type = 'all_shocks'
main(root_dir = root_dir,timestamps_path = timestamps_root_path,  dir_to_save= dir_to_save,compute_DM = True, contrast_type = 'all_shocks')



