import pandas as pd
import numpy as np
import os

                                         
#/////////////////////////////////////////////////////////
root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test'
subject = 'APM_03_H1'

#variable utilisé comme paramètres de la fonction en guise de test
data_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii\APM_03_H1\02-Analgesia'
dir_anat = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\Analyses_conn\DATA_Desmateaux_Rainville\H1\APM_03_H1\01-MEMPRAGE\wms201301101300-0004-00001-000176-01.nii'
timestamps_path = r'C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\times_stamps\ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx'
DM_name = 'DM_HYPER_' + subject
dir_to_save = os.path.join(root_dir,subject)

#subj_path est pour ajouter des fichiers dans le dossier de chaque sujet
subj_path = os.path.join(root_dir,subject)


#////////////régresseurs de mouvements////////////////
#stockage du fichier des régresseurs de mouvements
path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_all_shock'

#for file in os.listdir(path):

file = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_glm_all_shocks\APM_09_H1\beta_map_APM_09_H1_HYPER_all_shocks.nii'
subj_name = 'beta_map_APM_09_H1_HYPER_all_shocks'
import nibabel as nib

from nilearn.image import concat_imgs, mean_img

#fmri_img = nib.load(r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock\APM_02_H2\APM_02_H2_HYPER_concat_fmri.nii')
#mean_img = mean_img(fmri_img)
anat_img = nib.load(r'C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\Analyse_PPI_psy3008\Avg_Anat\Avg_24subjs_T1.hdr')

#############################################
#############################################
#if file.startswith('beta'):

print('FILE')

import nibabel as nib
import os


img = nib.load(file)
print(type(img))
print(img.shape)
print(img)


from nilearn import plotting

#=========================
#INTERACTIVE PLOT IN HTLM
#=========================
html_view = plotting.view_img(img,bg_img = anat_img, threshold=2, vmax=4, cut_coords=[-42, -16, 52], title="shock")

html_view.open_in_browser()


#=========================
#REGULAR STATIC PLOT WITH DIFFERENT OPTION
#=========================

#plotting.plot_stat_map(
#                    img, bg_img=anat_img, threshold=3.0, display_mode='x',
#                    cut_coords=3, black_bg=True, title='shock')
#plotting.show()



