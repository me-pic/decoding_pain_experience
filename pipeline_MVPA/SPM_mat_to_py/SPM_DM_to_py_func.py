import os
import pandas as pd
import glob 
from nilearn.image import concat_imgs
import nibabel as nib

def copy_folder_struct(origin_path, dir_copy):#creates folder in out_dir replicating the SPM_files_path structure
    """
    Arguments
    --------

    create_dir : Bool
                Ture by default. If True a folder structure with the same organization as SPM_files_path will be create in out_dir,
                e.g. subject01, sub02, sub. ... will be copied in the saving directory

    Example
    -------

    Apply this script to replicate folder structure into new one. E.g. origin_path contains subjects' folders, the same subjects' folders will be created at dir_copy path
    """

    for folder in os.listdir(origin_path):
        if 'APM' in folder:

            os.mkdir(os.path.join(dir_copy,folder))

#copy_folder_struct(origin_path = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\neut_shocks', dir_copy = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_all_runs')

def change_events_name(matrix,str_to_keep):

    for col in matrix.columns:
        if str_to_keep in col:
            new_col = col[6:-6]
            matrix.rename(columns = {col:new_col}, inplace = True)

    return matrix

def SPM_DM_to_py(SPM_files_path, out_dir):

    """
    Arguments
    --------

    SPM_files_path : str
                    path to folder containing subjects' files with event.csv and DPM_design_matrix.csv  
    out_dir : str
            Path where results will be saved

    Example
    -------

    This function was created to adapt design matrices from the SPM 'SPM.mat' file and import it in python structure. 
    The matlab script 'SPM_MAT_to_csv.m' should be applied prior to this one in order to save the SPM.mat structure as CSV. The path to those csv files should be 
        input as SPM_files_path.
    """

    for subject in os.listdir(SPM_files_path):

        #SPM dm and event extraction
        subj_path = os.path.join(SPM_files_path,subject)
        events = (pd.read_csv(os.path.join(subj_path, os.listdir(subj_path)[1]))) #read event, should change [index] according to folder
        print(events)
        #change events names
        changed_events = change_events_name(events,'shocks')
        print(changed_events)

        design_matrix= pd.DataFrame(pd.read_csv(os.path.join(subj_path, os.listdir(subj_path)[0]),header=None))#read DM.csv
        design_matrix.columns = changed_events.columns
        print(design_matrix.columns)
        #extract and concat fmri data files
        #subj_data_path = os.path.join(nii_path,subject)
        #subj_volumes = glob.glob(os.path.join(subj_data_path,'*','sw*')) #change the prefix 'sw*' according to .nii file names
        #fmri_time_series = concat_imgs(subj_volumes)

        print(design_matrix.shape)
        #save
        DM_name = 'DM_SPM_Hyper_Hypo.csv'
        #design_matrix.to_csv(os.path.join(out_dir,subject, DM_name))
        fmri_name = 'fmri_4D_concat_all_runs'
        #nib.save(fmri_time_series, os.path.join(save_path,subject,fmri_name))
        print('done saving for ' + subject)
        print('------------')
 
#calling function
SPM_dir = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_csv'
out = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\SPM_DM_single_event_all_runs'

SPM_DM_to_py(SPM_dir, out)
