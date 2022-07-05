
def get_subj_data(data_dir, prefix):

    #-------------Fonction description----------

    #Function that takes a directory of file and runs through all the file in it whilst it only list the path of the file starting
    #with the prefix given as argument

    #Example : in a directory you have many nii files, but you only want to use a sub-sample of them, e.g. the ones starting with the prefix swaf*

    #-----------------Variables-----------------
    #data_dir: path to a directory containing all files
    #prefix : prefix of the file of interest that you want to list the path

    #--------------------------------------------

    import os

    #Extraction des volumes pour un sujet dans une liste
    ls_volumes_all = os.listdir(data_dir)

    #Crée une liste avec seulement les fichiers commencant par sw
    swaf_list = [x for x in ls_volumes_all if x.startswith(prefix)]
    #print(swaf_list)

    #joindre le path avec les noms des volumes dans une liste
    #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
    subject_data = [os.path.join(data_dir, name) for name in swaf_list]


    return subject_data




def adapt_data(subj_path, current_file, subj_name, subj_data, timestamps_path_root):


    #-------------Fonction description----------

    #This function is quite specific to the data structure for which it has been created. It will return :
    # timestamps : a path to the timestamp that goes with a subject data,the design matrix name, under which,
    #it will later be saved, the mouvement regressor matrix, which will be used to create the design matrix
    #and finally the run, which is a string that will be used to keep track of the run that is computing, e.g. hyper or hypo.

    #-----------------Variables-----------------
    #current_file : we assume that this function will be used
    #while looping over different files that each contains the data
    #subj_path : path du sujet en cours, e.g. E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test\APM_02_H2
    #subj_name : nom du sujet, e.g. APM_02_H2
    #subj_data : a list with all the paths to nii volumes
    #to_save_root : path where you want to save results for each participant. Note that a file for each subject will be created if it doesn't exist
    #--------------------------------------------


    import numpy as np
    import os
    import pandas as pd
    from os.path import exists

    #///////////paramètres de mouvements/////////////////

    #stockage du fichier des régresseurs de mouvements
    #on veut dans notre cas le seul fichier dans subj_path qui commence par APM, car c'est le nom du fichier
    #contenant les régresseurs de mouvements

    movement_reg_sheet = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = subj_path + movement_reg_sheet[0]

    #On les stock dans une liste et on en fait l'extraction ensuite
    movement_reg_list = [i for i in os.listdir(subj_path) if i.startswith('APM')]
    mvmnt_reg_path = os.path.join(subj_path,movement_reg_list[0])

    #Read the file
    df_mvmnt_reg_full = pd.read_csv(mvmnt_reg_path, sep= '\s+', header=None)


    #////////////////////////////////////////////////////
    #encodage de la condition
    if subj_name.endswith('H2'):
        condition = 'H2'
    elif subj_name.endswith('H1'):
        condition = 'H1'

    else:
        condition = ''


    #//////////////////restriction des conditions et des runs/////////////////////

        #on veut contrôler si on se trouve dans la condition hyper ou hypo
        #Si on est dans hyper (02), on va définir le nom du path pour enregistrer la DM, le nom du fichier
        #Si on est en présence d'exception, soit pour les sujets : APM_02, APM_05, APM_17, APM_20, on va overwrite
        #les variables du path, du timestamps et du nom du fichier de la DM. Ces sujets ne sont que présent dans la
        #condition H2


    #02 = hyper et 03 = hypo
    if condition == 'H2':


        #on se trouve alors dans la condition HYPER
        if current_file.startswith('02') is True:


           #variables pour la suite

            DM_name = 'DM_HYPER_' + subj_name + '.csv'
            run = 'HYPER'

            #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

            #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
            #puisqu'on se trouve dans HYPER


            df_mvmnt_reg_HYPER = split_reg_lower(df_mvmnt_reg_full, len(subj_data))



            ##////////////TIMESTAMPS et RÉGRESSEURS DE MOUVEMENTS pour les exceptions//////////////
            if subj_name == 'APM_02_H2':
                timestamps = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx')

            elif subj_name == 'APM_05_H2':
                timestamps = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.xlsx')

            elif subj_name == 'APM_17_H2':
                timestamps =   os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.xlsx')

            elif subj_name == 'APM_20_H2':
                timestamps =   os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.xlsx')

            #timestamps HYPER pour les sujets normaux dans H2
            else :
                timestamps =   os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx')

            #/////////////return////////////////////
            return timestamps, DM_name, df_mvmnt_reg_HYPER, run


        #on et alors dans la condition 03-HYPO
        else :

            #variable pour la suite et timestamps

            DM_name = 'DM_HYPO_' + subj_name + '.csv'
            run = 'HYPO'

            #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

            #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
            #puisqu'on se trouve dans HYPER


            #from function_split_reg_mvmnt_hypo import split_reg_hypo

            df_mvmnt_reg_HYPO = split_reg_upper(df_mvmnt_reg_full, len(subj_data))

            #//////////////////////////////////

            if subj_name == 'APM_02_H2':
                timestamps =  os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.xlsx')

            elif subj_name == 'APM_05_H2':
                timestamps =   os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.xlsx')

            elif subj_name == 'APM_17_H2':
                timestamps =  os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.xlsx')

            elif subj_name == 'APM_20_H2':
                timestamps = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.xlsx')

            #timestamps HYPO/ANA pour les sujets normaux dans H2
            else :
                timestamps = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx')

            ##/////////////////////////////////
            return timestamps, DM_name, df_mvmnt_reg_HYPO, run


    ## HYPO si on se trouve dans le fichier 03-analgesia
    elif condition == 'H1':
        #print('ON EST ENTRÉ DANS LE ELIF DE LA CONDITION H1')


        if current_file.startswith('02') is True:

            #print('la condition est _' + condition + '_ et on est dans HYPO, 02')

            #Variables pour la suite et timestamps

            DM_name = 'DM_HYPO_' + subj_name + '.csv'
            run = 'HYPO'

            #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\




            df_mvmnt_reg_HYPO = split_reg_upper(df_mvmnt_reg_full, len(subj_data))

            #timestamps HYPO pour les sujets de H1
            timestamps =  os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx')

            ##/////////////////////////////////
            return timestamps, DM_name, df_mvmnt_reg_HYPO, run


        #Alors on est dans la condition 03-HYPER
        else:

            #print('la condition est _' + condition + '_ et on est dans HYPER 03')

            #Variables pour la suite et timestamps

            DM_name = 'DM_HYPER_' + subj_name + '.csv'
            run = 'HYPER'

            #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

            #we want to extract the lower part of the mouvement regressor matrix since we are in hyper run

            df_mvmnt_reg_HYPER = split_reg_lower(df_mvmnt_reg_full, len(subj_data))


            #timestamps HYPER pour les sujets de H1
            timestamps =  os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx')

            ##/////////////////////////////////
            return timestamps, DM_name, df_mvmnt_reg_HYPER, run

    else:

        print('le fichier en cours n\'est pas un sujet appartenant à H1 ou H2')




def split_reg_upper(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split)selon le nombre de volumes qu'on donne en argument (target_lenght)
    #-------------Fonction description----------

    #function that takes a matrix and split it horizontally at the index given as argument (target_lenght). Returns the **UPPER** part
    #of where the matrix was split.

    #-----------------Variables-----------------
    #matrix_to_split : a matrix
    #target_lenght : the index at which you want the split matrix
    #--------------------------------------------

     #on split horizontalement la matrice en une tranche de
     split_matrix = matrix_to_split.iloc[0:target_lenght, :]

     return split_matrix


def split_reg_lower(matrix_to_split, target_lenght):

    #funciton qui split une matrice (matrix_to_split) selon le nombre de volumes qu'on donne en argument (target_lenght)
    #-------------Fonction description----------

    #function that takes a matrix and split it horizontally at the index given as argument (target_lenght). Returns the **LOWER** part
    #of where the matrix was split.

    #-----------------Variables-----------------
    #matrix_to_split : a matrix
    #target_lenght : the index at which you want the split matrix
    #--------------------------------------------


    inverse_lenght = len(matrix_to_split) - target_lenght

    split_matrix = matrix_to_split.iloc[inverse_lenght :len(matrix_to_split), :]

    return split_matrix


