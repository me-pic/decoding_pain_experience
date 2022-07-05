import numpy as np
import os
import pandas as pd
import nibabel as nib
#pseudocode


#////////////////variables to set/////////////////////////
# /!\
#/_!_\ to change according to computer

#main path to data, change according to environment
root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\result_GLM\result_1st_level_all_shocks'
dir_to_save = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\result_GLM\result_2nd_level_all_shocks'

#SERVEUR elm

#root_dir = r'/data/rainville/dylan_projet_ivado_decodage/results_glm_all_shocks'
#dir_to_save = r'/data/rainville/dylan_projet_ivado_decodage/results/GLM_2nd_level_all_shocks'

#==============================================================
#store all subject's name in a list
ls_subj_name = [subject for subject in os.listdir(root_dir)]
print('Making second level analysis for subjects : {} '.format(ls_subj_name))

#############
#stocking subjects' name with the run associated with the beta map
#This variable will be used in the plotting section
ls_name_beta = []
for items in ls_subj_name:

    entry1 = items + '_hyper'
    entry2 = items + '_hypo'
    ls_name_beta.append(entry1)
    ls_name_beta.append(entry2)


#############
#make a list for all the subject's path to data
ls_subj_path  = [os.path.join(root_dir,subject) for subject in os.listdir(root_dir)]



#######################
#EXTRACTING DATA INPUT
#######################

#Empty list to list the paths to data
ls_data_input = []

#For eache subject, we add the beta maps paths tp a list
for subj_name in os.listdir(root_dir):

    #############
    #creating a path to acces data for the ungoing subject
    subj_data_path = os.path.join(root_dir,subj_name)

    #############
    #make a list of all the beta maps' paths

    #A function that takes a path and list all the paths' file that starts with a prefix
    #In our case, we want for each subject the path of the beta maps
    from A_data_prep import function_code as A

    #listing the paths to the maps for the ongoing subject
    ls_data_input.append(A.get_subj_data(subj_data_path, 'beta'))

    #As the previous function returns many lists, e.g. [[1,2],[3,4]..] in data_input, we want to
    #have only one list, e.g. [1,2,3,4]
    data_input = []

    for element in ls_data_input:

        data_input += element
    #############

#######################
#PLOTTING input data
#######################

from nilearn import plotting
import matplotlib.pyplot as plt

"""
fig, axes = plt.subplots(nrows=4, ncols=4)
for idx, indiv_beta_map in enumerate(data_input):
    plotting.plot_glass_brain(indiv_beta_map, colorbar=False, threshold=2.0,
                              title=ls_name_beta[idx],
                              plot_abs=False, display_mode='z')
fig.suptitle('subjects beta maps for each run of shocks')
plt.show()
"""
##########
#HTML INTERACTIVE PLOTTING
##########

#anat_img = nib.load(r'C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\Analyse_PPI_psy3008\Avg_Anat\Avg_24subjs_T1.hdr')

#count_idx = 0
#for indiv_beta_map in data_input:

    #loadinf nii file
    #beta_map = nib.load(indiv_beta_map)

        #plotting in HTML
    #html_view = plotting.view_img(beta_map,bg_img = anat_img, threshold=2, vmax=4, cut_coords=[-42, -16, 52], title=ls_name_beta[count_idx])

    #html_view.open_in_browser()

    #count_idx += 1



#######################
#DESIGN MATRIX
#######################


design_matrix = pd.DataFrame([1] * len(data_input),
                             columns=['intercept'])


#######################
#SECOND LEVEL MODEL
#######################

from nilearn.glm.second_level import SecondLevelModel

second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(data_input,
                                            design_matrix=design_matrix)

beta_map = second_level_model.compute_contrast(output_type='z_score')

#######################
#saving beta map

import nibabel as nib

beta_map_name = 'beta_map_2nd_level.nii'
nib.save(beta_map, os.path.join(dir_to_save, beta_map_name))


#######################
#PLOTTING ALL BETA MAPS
#######################

from scipy.stats import norm
p_val = 0.001
p001_unc = norm.isf(p_val)
display = plotting.plot_glass_brain(
    beta_map, threshold=p001_unc, colorbar=True, display_mode='z', plot_abs=False,
    title='beta map for each run of shocks (unc p<0.001)')
plotting.show()

#########
#HTLM
anat_img = nib.load(r'C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\Analyse_PPI_psy3008\Avg_Anat\Avg_24subjs_T1.hdr')

html_view = plotting.view_img(beta_map,bg_img = anat_img, threshold=2, vmax=4, cut_coords=[-42, -16, 52], title='avg beta map')

html_view.open_in_browser()
#########

import numpy as np
from nilearn.image import get_data, math_img

p_val = second_level_model.compute_contrast(output_type='p_value')
n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
# Correcting the p-values for multiple testing and taking negative logarithm
neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))"
                        .format(str(n_voxels)),
                        img=p_val)

cut_coords = [0]
# Since we are plotting negative log p-values and using a threshold equal to 1,
# it corresponds to corrected p-values lower than 10%, meaning that there is
# less than 10% probability to make a single false discovery (90% chance that
# we make no false discovery at all).  This threshold is much more conservative
# than the previous one.
threshold = 1
title = ('Group left-right button press: \n'
         'parametric test (FWER < 10%)')
display = plotting.plot_glass_brain(
    neg_log_pval, colorbar=True, display_mode='z', plot_abs=False, vmax=3,
    cut_coords=cut_coords, threshold=threshold, title=title)
plotting.show()

