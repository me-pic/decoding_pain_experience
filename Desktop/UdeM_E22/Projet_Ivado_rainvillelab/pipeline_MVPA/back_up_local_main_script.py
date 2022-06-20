import numpy as np
import os
import pandas as pd

#////////////////variables to set/////////////////////////
# /!\
#/_!_\ to change according to computer

#main path to data, change according to environment
root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii'
dir_to_save = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock'

#SERVEUR elm

#root_dir = r'/data/rainville/dylan_projet_ivado_decodage/Nii'
#dir_to_save = r'/data/rainville/dylan_projet_ivado_decodage/results_glm_1level'

#==============================================================
#store all subject's name in a list
ls_subj_name = [subject for subject in os.listdir(root_dir)]

#make a list for all the subject's path to data
ls_subj_path  = [os.path.join(root_dir,subject) for subject in os.listdir(root_dir)]


#////////////////iterations over files and data/////////////////////////

index_count = 0

#pour pour faire itération de chaque sujet à travers la liste de tous les path des sujets
for subj_path in ls_subj_path:

    #store subject's name
    subj_name = ls_subj_name[index_count]

    #for all the file in subject_path that starts with 02 or 03
    for file in [i for i in os.listdir(subj_path) if i.startswith('02') or i.startswith('03')]:

        data_path = os.path.join(subj_path,file)
        #print(file)

        #//////data//////
        #extract the nii data that is specific to each 02 or 03 file for a particular subject

        # /!\
        #/_!_\ to change according to computer
        #local
        timestamps_path_root = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER'

        #elm
        #timestamps_path_root = r'/data/rainville/dylan_projet_ivado_decodage/All_run_timestamps'

        from A_data_selection import function_get_subj_data

        #function that returns a list with all nii paths in the file
        subj_data = function_get_subj_data.get_subj_data(data_path, 'sw')

        #//////extraction of customize variable//////

        #here we extract some variable needed to make the design_matrix
        #Those variables change according to the condition (H2/H1) and the run (hyper/hypo)
        from manip_data_dev import adapt_data

        timestamps_path, DM_name, df_mvmnt_reg, condition = adapt_data(file, subj_path, subj_name, subj_data, timestamps_path_root, dir_to_save)

        #print(timestamps_path)
        #print(DM_name)
        #print(df_mvmnt_reg.shape)

        ##############################################
        #check if the DM already exists in path to we save computing time

        if os.path.exists(os.path.join(dir_to_save, subj_name, DM_name)) is False:

            ###############################################
            #//////DESIGN MATRIX//////
            ###############################################

            #In this section, we reuse the variable defined before to make the DM
            from function_DM import create_DM

            print('====================================')
            print('COMPUTING design matrix under name : ' + DM_name + ' for subject ' + subj_name)

            design_matrix, fmri_img= create_DM(subj_data, timestamps_path, DM_name, df_mvmnt_reg)

            #/////prints for info/////

            print('SHAPE DES REGRESSEURS DE MOUVEMENT : {} '.format(df_mvmnt_reg.shape))
            print('SHAPE DES VOLUMES CONCATÉNÉS NII : {} '.format(fmri_img.shape))
            print('SHAPE DE LA DESIGN MATRIX : {} '.format(design_matrix.shape))

            #//////////SAVING OUTPUTS/////////

            #saving design_matrix
            design_matrix.to_csv(os.path.join(dir_to_save,subj_name,DM_name), index = False)

            #saving concatenated nii files

            import nibabel as nib
            #nib.save(design_matrix, os.path.join(dir_to_save,subj_name,DM_name))

            fmri_img_name = subj_name + '_' + condition + '_' + 'concat_fmri.nii'
            nib.save(fmri_img, os.path.join(dir_to_save,subj_name,fmri_img_name))
            #print('SAVING fmri_img having SHAPE : {} '.format(fmri_img.shape))

        #if the design matrix is already existant in the folder
        else:

            print('Design matrix in condition _{}_  is already existant for : {} '.format(condition, subj_name))

            design_matrix = pd.read_csv(os.path.join(dir_to_save, subj_name, DM_name))

            import nibabel as nib

            fmri_img_name = subj_name + '_' + condition + '_' + 'concat_fmri.nii'

            #we want to select only the file nii that starts with AMP
            #target_file = [x for x in os.listdir(os.path.join(dir_to_save,subj_name)) if x.startswith('APM') and if x.]

            fmri_img = nib.load(os.path.join(dir_to_save,subj_name, fmri_img_name))
                #join(dir_to_save, subj_name, fmri_img_name))


        #//////Plot option//////

        #Uncomment to plot the design matrix as it's generated
        #from nilearn.plotting import plot_design_matrix

        #from nilearn.plotting import plot_design_matrix
        #import matplotlib.pyplot as plt

        #plot_design_matrix(design_matrix)
        #plt.show()



        ################################################
        #///////CONTRAST////////
        ################################################

        #define a done_file name to check if it  already exists in file
        done_file_name = 'done_contrast_' + condition + '.txt'

        #Verify if contrast for both design matrices has been donne to avoid unecessary computing
        if os.path.exists(os.path.join(dir_to_save,subj_name,done_file_name)) == False:

            print('===================================')
            print('STARTING CONTRAST FOR : ' + subj_name)

            ###################
            #COMPUTING CONTRAST
            from glm_make_contrast_1shock import glm_contrast_1event

            #on met en argument, la DM, le path pour save, le nom du sujet et les volumes nii
            glm_contrast_1event(design_matrix,os.path.join(dir_to_save,subj_name), subj_name, fmri_img)

            ###################
            #SAVING AND KEEPING TRACK OF WHAT HAS BEEN DONE

            #write done_contrast_hyper/hypo to keep track of what has been computed
            done_file=open(os.path.join(dir_to_save,subj_name,done_file_name), 'w')
            done_file.write('')
            done_file.close()
            print('HAVE WRITTEN : ' + done_file_name)


        else:
            print('Contrast : _{}_ has already been done for subject : {} '.format(condition, subj_name))



    index_count += 1


