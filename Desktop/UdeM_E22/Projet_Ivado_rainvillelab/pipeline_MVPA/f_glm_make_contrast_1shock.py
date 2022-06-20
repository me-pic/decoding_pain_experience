
#design_matrix = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test\APM_03_H1\DM_HYPER_APM_03_H1.npy'
import pandas as pd
import os
import numpy as np


def glm_contrast_1event(design_matrix, subj_result_path, subj_name, fmri_img):


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
    identity_matrix = np.eye(design_matrix.shape[1],design_matrix.shape[1])
    null_matrix = np.zeros(identity_matrix.shape)
    #print(null_matrix.shape), print('NULL MATRIX')

    #Création d'un dict ayant comme clé les col de la design_matrix et comme valeur 0 ou 1
    none_contrasts = dict([(column, null_matrix[i])
      for i, column in enumerate(design_matrix.columns)])
    #print(none_contrasts)

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
