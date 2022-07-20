
#design_matrix = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test\APM_03_H1\DM_HYPER_APM_03_H1.npy'
import pandas as pd
import os
import numpy as np
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import concat_imgs, mean_img
import nibabel as nib

def glm_contrast_delete_event(design_matrix, subj_result_path, subj_name, fmri_img):


    #////////////////Importer les donneés nifti///////////////////////////
    from nilearn.image import concat_imgs, mean_img
    import nibabel as nib

    fmri_img_name = subj_name + '_concat_fmri.nii'
    #print('fmri.shape : {} '.format(fmri_img.shape))


    from nilearn.image import mean_img

    #template to project the statistical map
    #anat_img = nib.load('C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\Avg_Anat')
    #mean_img = mean_img(fmri_img)

    #####################################################
    #####################################################

    #///////////MODÈLE/////////////////

    print('====================================')
    print('COMPUTING GLM for subject ' + subj_name)

    from nilearn.glm.first_level import FirstLevelModel

    fmri_glm = FirstLevelModel(t_r=3, #ok
                               noise_model='ar1',
                               standardize=False,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass=.00233645)

    fmri_glm = fmri_glm.fit(fmri_img, design_matrices = design_matrix)

    #/////////////basic matrices for contrast//////////////////

    #matrices nulle et identité pour faire les contrastes pour une col
    #de la design_matrix à la fois
    identity_matrix = np.eye(design_matrix.shape[1])
    null_matrix = np.zeros(identity_matrix.shape)
    #print(null_matrix.shape), print('NULL MATRIX')

    #Création d'un dict ayant comme clé les col de la design_matrix et comme valeur 0 ou 1
    none_contrasts = dict([(column, np.zeros(identity_matrix.shape[1]))
      for column in design_matrix.columns])


    #contrast for each shock and save it under result_glm_1shock
    #///////////////contrast loop///////////////////////////
    #boucle qui prend la colonne dans le eye à l'index de [Nieme titre de la
    #design_matrix qui contient shock] et qui la met dans un dict

    #faire un dict avec le même nombre de col que la design_matrix. mais remplie de 0
    #cherche dans la design_matrix pour l'index de la col qui contient 'shock'
    #si oui :
        #stocker son index, aller chercher cet index dans la matrice eye

    indx = 0
    for keys in none_contrasts:


        if type(keys) is str:
            #change according to interst
            string_interest = 'shock'

            #liste des clés pour retrouver l'index de la colonne d'intérêt
            key_list = list(none_contrasts)

            #si la clé du dict contient le mot shock
            if string_interest in keys:

                #///////////preparation contraste///////////

                #aller chercher le nom de la colonne en cours
                actual_key_name = key_list[indx]


                #aller chercher la colonne de contraste d'intérêt à la bonne
                #position dans identity_matrix
                contrast_col = identity_matrix[:, indx]
                #print('La colonne qui a été importée est  {} '.format(contrast_col))

                #mettre une colonne de contraste dans cette clé
                none_contrasts[actual_key_name] = contrast_col
                #print(none_contrasts)

                #///////////computing contrast///////////
                print('==================================')
                print('COMPUTING CONTRAST FOR : {} '.format(actual_key_name))


                from nilearn import plotting

                # compute the contrasts
                beta_map = fmri_glm.compute_contrast(
                    contrast_col, output_type='z_score')
                print('Will save beta_map as a : {} , having shape : {} '.format(type(beta_map),beta_map.shape))


                #//////////////Plot option////////////////

                # plot the contrasts as soon as they're generated
                # the display is overlaid on the mean fMRI image
                # a threshold of 3.0 is used, more sophisticated choices are possible
                #plotting.plot_stat_map(
                    #beta_map, bg_img=mean_img, threshold=3.0, display_mode='z',
                    #cut_coords=3, black_bg=True, title=actual_key_name)

                #plotting.show()


                #////////SAVING OUTPUT//////////////
                import nibabel as nib

                name_to_save = 'beta_map_' + subj_name + '_' + actual_key_name

                #save la structure nifti, qui est la carte d'activation dans le dossier résultat/participant x
                nib.save(beta_map, os.path.join(subj_result_path, name_to_save))
        else:
            print('KEY IS NOT A STR')

        indx += 1



            #return beta_map

#=====================================
#CONTRAST ALL SHOCKS
#=====================================


####manual test#######
#design_matrix = pd.read_csv(r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock\APM_02_H2\DM_HYPER_APM_02_H2.csv')
#subj_result_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_mean_shock'
#subj_name = 'APM_02_H2'
#fmri_img = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\result_glm_1shock\APM_02_H2\APM_02_H2_HYPER_concat_fmri.nii'
######################

def glm_contrast_1event(design_matrix, subj_result_path, subj_name,fmri_img, run = None):

    #====================
    #Function that takes a design matrix, a path to save, a subject name, a run name and a 4D nii file to conpute contrast
    #This function is supposed to be used in a loop over many subject file. Therefor, it has as arguments a path to save, a subject's
    #name and a run name to save the file under a name that will take into account such information.

    #Arguments :
    #design_matrix
    #subj_result_path : a path where you want to save the contrast
    #subj_name : name of the subject
    #fmri_img : a 4D nii file containing data
    #run : *optionnal. A string containing the name of the actual run that will eb added to the name of the saved contrast file.
        #If not specified, no run string will be included in the saved contrast name



    #////////////////Importer les donneés nifti///////////////////////////

    fmri_img_name = subj_name + '_concat_fmri.nii'
    #print('fmri.shape : {} '.format(fmri_img.shape))
    from nilearn.image import mean_img

    #///////////MODÈLE/////////////////
    print('==============')
    print('COMPUTING GLM for subject ' + subj_name)

    fmri_glm = FirstLevelModel(t_r=3, #ok
                               noise_model='ar1',
                               standardize=False,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass=.00233645)
    fmri_glm.fit(fmri_img, design_matrices = design_matrix)

    #==============
    #identity matrix having shape of number of regressor x number of regressor in the Design matrix
    #Each column will serve to encode a 1 in a specifi columns of interest
    #==============
    #none contrast is a dictionnary having design matrix column name as key name. The values of each
    #vector is a 'number of regressor'(design_matrix.shape[1]) long.
    #the vector in each key is 0 or 1 and serve to encode the contrast in further steps

    identity_matrix = np.eye(design_matrix.shape[1],design_matrix.shape[1])
    null_matrix = np.zeros(identity_matrix.shape)
    #print(null_matrix.shape), print('NULL MATRIX')

    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrix.columns)])
    #print('NONE CONTRAST  {}'.format(none_contrasts))
    print('lenght of NONE CONTRAST  {}'.format(len(none_contrasts)))

    contrast_vector = np.zeros((design_matrix.shape[1]))#ones will be added to this vector as we specify which regressor we want to contrast

    #list of all the regressors/keys to keep track of the regressors we've added to contrast
    key_list = list(none_contrasts)

    #======================================
    #boucle qui prend la colonne dans identity_matrix à l'index de [Nieme titre de la
    #design_matrix qui contient shock, e.g. N_ANA_shock_x] et qui la met dans un dict
    #cherche dans la design_matrix pour l'index de la col qui contient 'shock'
    #si oui :
        #stocker son index, aller chercher cet index dans la matrice eye
    indx = 0
    for keys in none_contrasts:

        #if the key is str, in order to exclude the drifts and other parameters
        if type(keys) is str:

            string_interest = 'shock' #In this case every key that has the string 'shock' will be attributed a 1 in that column

            if string_interest in keys:#if the keys contains the word 'shock'
                actual_key_name = keys
                contrast_vector = identity_matrix[:, indx]

                #--------contrast---------
                print('Contrasting for : ', actual_key_name)
                print('==============')
                print('COMPUTING CONTRAST')
                #print('With the CONTRAST VECTOR : {} '.format(contrast_vector))
                beta_map = fmri_glm.compute_contrast(
                     contrast_vector, output_type='z_score')#compute the contrasts with contrast vector

                #---------Plot option----------
                #plotting.plot_stat_map(
                    #beta_map, threshold=3.0, display_mode='z',
                    #cut_coords=3, black_bg=True, title=actual_key_name)
                #plotting.show()

                #---------SAVING OUTPUT--------
                if run == None:#control if a run string has been provided to include in the name of the file to save
                    name_to_save = 'beta_map_' + subj_name  + '_' + actual_key_name
                    #save la structure nifti, qui est la carte d'activation dans le dossier résultat/participant x
                    nib.save(beta_map, os.path.join(subj_result_path, name_to_save))

                if run != None:
                    name_to_save = 'beta_map_' + subj_name + '_' + run + '_' + actual_key_name
                    #save la structure nifti, qui est la carte d'activation dans le dossier résultat/participant x
                    nib.save(beta_map, os.path.join(subj_result_path, name_to_save))
                print('Have saved beta_map as a : {} , having shape : {} , under name : {}'.format(type(beta_map),beta_map.shape, name_to_save))

        indx += 1


















    return beta_map

def glm_contrast_all_shocks(design_matrix, subj_result_path, subj_name,fmri_img, run = None):

    #====================
    #Function that takes a design matrix, a path to save, a subject name, a run name and a 4D nii file to conpute contrast
    #This function is supposed to be used in a loop over many subject file. Therefor, it has as arguments a path to save, a subject's
    #name and a run name to save the file under a name that will take into account such information.

    #Arguments :
    #design_matrix
    #subj_result_path : a path where you want to save the contrast
    #subj_name : name of the subject
    #fmri_img : a 4D nii file containing data
    #run : *optionnal. A string containing the name of the actual run that will eb added to the name of the saved contrast file.
        #If not specified, no run string will be included in the saved contrast name



    #////////////////Importer les donneés nifti///////////////////////////

    fmri_img_name = subj_name + '_concat_fmri.nii'
    #print('fmri.shape : {} '.format(fmri_img.shape))
    from nilearn.image import mean_img

    #///////////MODÈLE/////////////////
    print('==============')
    print('COMPUTING GLM for subject ' + subj_name)

    fmri_glm = FirstLevelModel(t_r=3, #ok
                               noise_model='ar1',
                               standardize=False,
                               hrf_model='spm',
                               drift_model='cosine',
                               high_pass=.00233645)
    print(fmri_glm)
    fmri_glm.fit(fmri_img, design_matrices = design_matrix)

    #==============
    #identity matrix having shape of number of regressor x number of regressor in the Design matrix
    #Each column will serve to encode a 1 in a specifi columns of interest
    #==============
    #none contrast is a dictionnary having design matrix column name as key name. The values of each
    #vector is a 'number of regressor'(design_matrix.shape[1]) long.
    #the vector in each key is 0 or 1 and serve to encode the contrast in further steps

    identity_matrix = np.eye(design_matrix.shape[1],design_matrix.shape[1])
    null_matrix = np.zeros(identity_matrix.shape)
    #print(null_matrix.shape), print('NULL MATRIX')

    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrix.columns)])
    #print('NONE CONTRAST  {}'.format(none_contrasts))
    print('lenght of NONE CONTRAST  {}'.format(len(none_contrasts)))

    contrast_vector = np.zeros((design_matrix.shape[1]))#ones will be added to this vector as we specify which regressor we want to contrast

    #list of all the regressors/keys to keep track of the regressors we've added to contrast
    ls_keys = []
    key_list = list(none_contrasts)

    #======================================
    #boucle qui prend la colonne dans identity_matrix à l'index de [Nieme titre de la
    #design_matrix qui contient shock, e.g. N_ANA_shock_x] et qui la met dans un dict
    #cherche dans la design_matrix pour l'index de la col qui contient 'shock'
    #si oui :
        #stocker son index, aller chercher cet index dans la matrice eye
    indx = 0
    for keys in none_contrasts:

        #if the key is str, in order to exclude the drifts and other parameters
        if type(keys) is str:

            string_interest = 'shock' #In this case every key that has the string 'shock' will be attributed a 1 in that column

            if string_interest in keys:#if the keys contains the word 'shock'
                actual_key_name = key_list[indx]
                ls_keys.append(keys)#extract the actual regressor/key name with the index position
                contrast_vector += identity_matrix[:, indx] #sum of the contrast vector with the identity matrix column to stack the ones
                                                            #in the contrast vector for all regressors of interest
        indx += 1

    print(ls_keys, ' : All the keys added to contrast')

    #--------computing contrast---------
    print('==============')
    print('COMPUTING CONTRAST')
    print('With the CONTRAST VECTOR : {} '.format(contrast_vector))
    print('For the following REGRESSORS: {} '.format(ls_keys))

    beta_map = fmri_glm.compute_contrast(
         contrast_vector, output_type='z_score')#compute the contrasts with contrast vector

     #---------Plot option----------
    plotting.plot_stat_map(
        beta_map, threshold=3.0, display_mode='z',
        cut_coords=3, black_bg=True, title=actual_key_name)
    plotting.show()

     #////////SAVING OUTPUT//////////////
    if run == None:#control if a run string has been provided to include in the name of the file to save

        name_to_save = 'beta_map_' + subj_name  + '_all_shocks'
        #save la structure nifti, qui est la carte d'activation dans le dossier résultat/participant x
        nib.save(beta_map, os.path.join(subj_result_path, name_to_save))

    if run != None:

        name_to_save = 'beta_map_' + subj_name + '_' + run + '_all_shocks'
        #save la structure nifti, qui est la carte d'activation dans le dossier résultat/participant x
        nib.save(beta_map, os.path.join(subj_result_path, name_to_save))

    print('Have saved beta_map as a : {} , having shape : {} , under name : {}'.format(type(beta_map),beta_map.shape, name_to_save))

    return beta_map


