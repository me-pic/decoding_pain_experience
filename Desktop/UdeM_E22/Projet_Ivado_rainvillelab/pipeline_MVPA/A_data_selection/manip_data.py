import numpy as np
import os

import function_DM
import pandas as pd

from os.path import exists
#path principal vers les données
path = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii'


for subject in os.listdir(path):


    #///////////DIR///////////////////////////////
    #subj_path est pour ajouter des fichiers dans le dossier de chaque sujet
    subj_path = os.path.join(path,subject)
    #stockage du fichier des régresseurs de mouvements
    movement_reg_sheet = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = subj_path + movement_reg_sheet[0]
    #print(mvmnt_reg_path + ' : CHEMIN DU FICHIER DES RÉGRESSEURS DE MOUVEMENTS')

    
    #///////////paramètres de mouvements/////////////////

    #on veut dans notre cas le seul fichier dans subj_path qui commence par APM, car c'est le nom du fichier
    #contenant les régresseurs de mouvements

    #On les stock dans une liste et on en fait l'extraction ensuite
    movement_reg_list = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = os.path.join(subj_path,movement_reg_list[0])

    #Read the file
    df_mvmnt_reg_full = pd.read_csv(mvmnt_reg_path, sep= '\s+', header=None)

    #*******half_lenght_df_reg = int(len(df_mvmnt_reg_full)/2)


    #////////////////////////////////////////////////////
    #encodage de la condition

    if subject.endswith('H2'):
        condition = 'H2'
    elif subject.endswith('H1'):
        condition = 'H1'
        print('on est dans la condition H1')
    else:
        condition = ''

    #////////////////PREP TO SAVE/////////////////////////

    to_save_root = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\pipeline_MVPA\results_glm_1level'

    if exists(os.path.join(to_save_root,subject)) == False:

                    os.mkdir(os.path.join(to_save_root,subject))


    #//////////////////Boucle principale/////////////////////

    for file in [i for i in os.listdir(subj_path) if i.startswith('02') or i.startswith('03')]:


        print(file + ' : FICHIER 02 ou 03 en cours')

        #on veut contrôler si on se trouve dans la condition hyper ou hypo
        #Si on est dans hyper (02), on va définir le nom du path pour enregistrer la DM, le nom du fichier
        #Si on est en présence d'exception, soit pour les sujets : APM_02, APM_05, APM_17, APM_20, on va overwrite
        #les variables du path, du timestamps et du nom du fichier de la DM. Ces sujets ne sont que présent dans la
        #condition H2


        ##HYPER, donc 02 = hyper et 03 = hypo
        if condition == 'H2':


            #on se trouve alors dans la condition HYPER
            if file.startswith('02') is True:
                print('la condition est _' + condition + '_ et on est dans hyper, 02')

               #variables pour la suite
                data_dir_all_volumes = os.path.join(subj_path,file) #contient les volumes
                DM_name = 'DM_HYPER_' + subject + '.csv'

                 #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

                #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
                #puisqu'on se trouve dans HYPER

                #appel de la fonction concat pour aller chercher la variable subject data
                from function_concat_fmri_img import concat_fmri_igm
                fmri_img, subject_data = concat_fmri_igm(data_dir_all_volumes, 'sw')

                from function_split_reg_mvmnt_hyper import split_reg_hyper

                df_mvmnt_reg_HYPER = split_reg_hyper(df_mvmnt_reg_full, len(subject_data))


                ##////////////TIMESTAMPS et RÉGRESSEURS DE MOUVEMENTS pour les exceptions//////////////
                if subject == 'APM_02_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx'

                elif subject == 'APM_05_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.xlsx'

                elif subject == 'APM_17_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.xlsx'

                elif subject == 'APM_20_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.xlsx'

                #timestamps HYPER pour les sujets normaux dans H2
                else :
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx'

                ##/////////////Design Matrix////////////////////
                from function_DM import create_DM
                design_matrix, fmri_img = create_DM(data_dir_all_volumes, timestamps, DM_name, df_mvmnt_reg_HYPER)

                #/////////////SAVING OUTPUT////////////////

                design_matrix.to_csv(os.path.join(to_save_root,subject,DM_name), index = False)

                import nibabel as nib
                #nib.save(design_matrix, os.path.join(to_save_root,subject,DM_name))

                fmri_img_name = subject + '_' + 'concat_fmri.nii'
                nib.save(fmri_img, os.path.join(to_save_root,subject,fmri_img_name))
                print('fmri.shape : {} times'.format(fmri_img.shape))


            #on et alors dans la condition 03-HYPO
            else :

                #variable pour la suite et timestamps
                data_dir_all_volumes = os.path.join(subj_path,file)
                DM_name = 'DM_HYPO_' + subject + '.csv'

                #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

                #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
                #puisqu'on se trouve dans HYPER

                #appel de la fonction concat pour aller chercher la variable subject data
                from function_concat_fmri_img import concat_fmri_igm
                fmri_img, subject_data = concat_fmri_igm(data_dir_all_volumes, 'sw')

                from function_split_reg_mvmnt_hypo import split_reg_hypo

                df_mvmnt_reg_HYPO = split_reg_hypo(df_mvmnt_reg_full, len(subject_data))

                #//////////////////////////////////

                if subject == 'APM_02_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.xlsx'

                elif subject == 'APM_05_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.xlsx'

                elif subject == 'APM_17_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.xlsx'

                elif subject == 'APM_20_H2':
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.xlsx'

                #timestamps HYPO/ANA pour les sujets normaux dans H2
                else :
                    timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx'

                ##/////////////////////////////////
                from function_DM import create_DM
                design_matrix, fmri_img = create_DM(data_dir_all_volumes, timestamps, DM_name, df_mvmnt_reg_HYPO)

                #/////////////SAVING OUTPUT////////////////bnm

                design_matrix.to_csv(os.path.join(to_save_root,subject,DM_name), index = False)

                import nibabel as nib
                #nib.save(design_matrix, os.path.join(to_save_root,subject,DM_name))

                fmri_img_name = subject + '_' + 'concat_fmri.nii'
                nib.save(fmri_img, os.path.join(to_save_root,subject,fmri_img_name))
                print('fmri.shape : {} times'.format(fmri_img.shape))
        ## HYPO si on se trouve dans le fichier 03-analgesia
        elif condition == 'H1':
            #print('ON EST ENTRÉ DANS LE ELIF DE LA CONDITION H1')


            if file.startswith('02') is True:
                print('la condition est _' + condition + '_ et on est dans HYPO, 02')

                #Variables pour la suite et timestamps
                data_dir_all_volumes = os.path.join(subj_path,file)
                DM_name = 'DM_HYPO_' + subject + '.csv'

                #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

                #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
                #puisqu'on se trouve dans HYPER

                #appel de la fonction concat pour aller chercher la variable subject data
                from function_concat_fmri_img import concat_fmri_igm
                fmri_img, subject_data = concat_fmri_igm(data_dir_all_volumes, 'sw')

                from function_split_reg_mvmnt_hypo import split_reg_hypo

                df_mvmnt_reg_HYPO = split_reg_hypo(df_mvmnt_reg_full, len(subject_data))

                #timestamps HYPO pour les sujets de H1
                timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPO\ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx'

                ##/////////////////////////////////
                from function_DM import create_DM
                design_matrix, fmri_img = create_DM(data_dir_all_volumes, timestamps, DM_name,df_mvmnt_reg_HYPO)
                print(type(design_matrix))

                #/////////////SAVING OUTPUT////////////////

                design_matrix.to_csv(os.path.join(to_save_root,subject,DM_name), index = False)

                import nibabel as nib

                #nib.save(design_matrix, os.path.join(to_save_root,subject,DM_name))

                fmri_img_name = subject + '_' + 'concat_fmri.nii'
                nib.save(fmri_img, os.path.join(to_save_root,subject,fmri_img_name))

            #Alors on est dans la condition 03-HYPER
            else:

                print('la condition est _' + condition + '_ et on est dans HYPER 03')

                #Variables pour la suite et timestamps
                data_dir_all_volumes = os.path.join(subj_path,file)
                DM_name = 'DM_HYPER_' + subject + '.csv'

                #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

                #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
                #puisqu'on se trouve dans HYPER

                #appel de la fonction concat pour aller chercher la variable subject data
                from function_concat_fmri_img import concat_fmri_igm
                fmri_img, subject_data = concat_fmri_igm(data_dir_all_volumes, 'sw')

                from function_split_reg_mvmnt_hyper import split_reg_hyper

                df_mvmnt_reg_HYPER = split_reg_hyper(df_mvmnt_reg_full, len(subject_data))


                #timestamps HYPER pour les sujets de H1
                timestamps = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\time_stamps_HYPER\ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx'

                ##/////////////////////////////////
                from function_DM import create_DM
                design_matrix, fmri_img = create_DM(data_dir_all_volumes, timestamps, DM_name, df_mvmnt_reg_HYPER)

                #/////////////SAVING OUTPUT////////////////

                design_matrix.to_csv(os.path.join(to_save_root,subject,DM_name), index = False)

                import nibabel as nib
                fmri_img_name = subject + '_' + 'concat_fmri.nii'
                nib.save(fmri_img, os.path.join(to_save_root,subject,fmri_img_name))
        else:

            print('le fichier en cours n\'est pas un sujet appartenant à H1 ou H2')






