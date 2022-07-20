
#////////////////variables to set/////////////////////////
# /!\
#/_!_\ to change according to computer

#main path to data, change according to environment
root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii'
dir_to_save = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\testing_scripts'

#SERVEUR elm

#root_dir = r'/data/rainville/dylan_projet_ivado_decodage/Nii'
#dir_to_save = r'/data/rainville/dylan_projet_ivado_decodage/results/GLM_1st_level_all_shocks'

 # /!\
 #/_!_\ to change according to computer
 #local
timestamps_root_path = r'C:\Users\Dylan\Desktop\BAC_neurocog\UM_H22\PSY3008\times_stamps'

 #elm
 #timestamps_path_root = r'/data/rainville/dylan_projet_ivado_decodage/All_run_timestamps'


import numpy as np
import os
import pandas as pd
import glob
import nibabel as nib
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
from A_data_prep import function_code as A_data_prep
from B_design_matrix import function_DM as B_design_matrix
from C_contrasts import function_contrasts as contrast

#==============================================================
#store all subject's name in a list
ls_subj_name = [subject for subject in os.listdir(root_dir)]

#make a list for all the subject's path to data
ls_subj_path  = [os.path.join(root_dir,subject) for subject in os.listdir(root_dir)]


for subj_path in ls_subj_path:

    #=================
    subj_name = os.path.basename(os.path.normpath(subj_path)) #extract last part of subj_path to get subject's name
    print(subj_name + ' = subj_name')
    #=================
    #Creating a dir to save, only if it doesn't exists
    if os.path.exists(os.path.join(dir_to_save,subj_name)) is False:

        os.mkdir(os.path.join(dir_to_save,subj_name))
    else :
        pass
    #=================
    #looking for the regressors' file, which starts with 'APM' and read it as csv
    movement_reg_name = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = os.path.join(subj_path,movement_reg_name[0])#movement_reg_name[0] because there's only one item in the list
    df_mvmnt_reg_full = pd.read_csv(mvmnt_reg_path, sep= '\s+', header=None)#full because we'll split it later according to condition

    #=================
    #file names that contains the fMRI volumes.
    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'
    test_index = 0
    #Main loop that will go over the condition's file, and generate a design matrix (DM) and a contrast according to the condition
    #In that loop, a Timestamps,a DM name,a mouvement regressors dataframe, a DM and statistical maps will be generated and saved
    for condition_file in [i for i in os.listdir(subj_path) if str_analgesia in i or str_hyper in i]:
        print(condition_file + ' = condition_file')

        #-------Extracting fMRI volumes-------

        data_path = os.path.join(subj_path,condition_file) #path to the data such as : /subj_01/02-Analgesia/<all nii files>
        subj_volumes= glob.glob(os.path.join(data_path,'sw*'))#extracting all the nii files that start with sw
        print('{} NII files in path {} for subject {}'.format(len(subj_volumes),data_path,subj_name))
        print('lenght of movement regesssor df : {}'.format(len(df_mvmnt_reg_full)))

        #-------Extracting timestamps--------

        timestamps = A_data_prep.get_timestamps(data_path, subj_name,timestamps_root_path,return_df =True)
        timestamps.sort_values(by=['onset'])

        #-------design matrix name and mouvement regessors--------

        #defining design matrix name and mouvement regessors according to condition
        if str_analgesia in condition_file:
            condition = 'HYPO'
            DM_name = 'DM_HYPO_' + subj_name + '.csv' #Initializing the name under which the design matrix will be saved
            #splitting either the first half or lower half of the mvmnt regressor df according to condition (analg/hyper)
            mvmnt_reg_df = A_data_prep.split_reg_upper(df_mvmnt_reg_full,len(subj_volumes))

        else:
            condition = 'HYPER'
            DM_name = 'DM_HYPER_' + subj_name + '.csv'
            mvmnt_reg_df = A_data_prep.split_reg_lower(df_mvmnt_reg_full,len(subj_volumes))

        #------DESIGN MATRIX------
        #check if the DM already exists in path to we save computing time
        if os.path.exists(os.path.join(dir_to_save, subj_name, DM_name)) is False:

            design_matrix, fmri_time_series = B_design_matrix.create_DM(subj_volumes, timestamps, DM_name, mvmnt_reg_df)

            #----------SAVING OUTPUTS------------
            #saving design_matrix and time series
            design_matrix.to_csv(os.path.join(dir_to_save,subj_name,DM_name), index = False)

            fmri_img_name = subj_name + '_' + condition + '_fmri_time_series.nii'
            nib.save(fmri_time_series, os.path.join(dir_to_save,subj_name,fmri_img_name))

        else:
            print('Design matrix in condition _{}_  is already existant for : {} '.format(condition, subj_name))

            design_matrix = pd.read_csv(os.path.join(dir_to_save, subj_name, DM_name))
            fmri_img_name = subj_name + '_' + condition + '_fmri_time_series.nii'
            print(os.path.join(dir_to_save,subj_name, fmri_img_name))
            fmri_time_series = nib.load(os.path.join(dir_to_save,subj_name, fmri_img_name))

        #-------Plot option-------
        #Uncomment to plot the design matrix as it's generated
        #from nilearn.plotting import plot_design_matrix
        #plot_design_matrix(design_matrix)
        #plt.show()

        #-------CONTRAST-------
        #define a done_file name to check if it  already exists in file
        done_file_name = 'done_contrast_' + condition + '.txt'

        #Verify if contrast for both design matrices has been donne to avoid unecessary computing
        if os.path.exists(os.path.join(dir_to_save,subj_name,done_file_name)) == False : #to enter the contrast section:
            #contrast for a single shock activation map
            contrast.glm_contrast_1event(design_matrix, os.path.join(dir_to_save,subj_name), subj_name, fmri_time_series, run = condition)

            #contrast for all shocks activation map, one for each design matrix will be made
            #contrast.glm_contrast_all_shocks(design_matrix, os.path.join(dir_to_save,subj_name), subj_name, fmri_time_series, run = condition)

            #-------SAVING-------
            #write done_contrast_hyper/hypo to keep track of what has been computed
            done_file=open(os.path.join(dir_to_save,subj_name,done_file_name), 'w')
            done_file.write('')
            done_file.close()
            print('HAVE WRITTEN : ' + done_file_name)

        else:
            print('Contrast : _{}_ has already been done for subject : {} '.format(condition, subj_name))
        test_index += 1

#Argument parser
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str) #dir to the subjects' files containing the fmri data
    parser.add_argument("--dir_to_save", type=str) #path to save the output
    parser.add_argument("--timestamps_path_root", type=str) #path to the timestamps files
    args = parser.parse_args()
