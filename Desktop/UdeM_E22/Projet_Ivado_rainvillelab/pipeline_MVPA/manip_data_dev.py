

def adapt_data(subj_path, current_file, subj_name, subj_data, timestamps_path_root):

    import numpy as np
    import os
    import function_DM
    import pandas as pd
    from os.path import exists

    #-----------------description fonction-----------------
    #current_file : we assume that a loop will go over some files containing datai
    #subj_path : path du sujet en cours, e.g. E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii_test\APM_02_H2
    #subj_name : nom du sujet, e.g. APM_02_H2
    #subj_data : a list with all the paths to nii volumes
    #to_save_root : path where you want to save results for each participant. Note that a file for each subject will be created if it doesn't exist
    #----------------------------------


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


    ##donc 02 = hyper et 03 = hypo
    if condition == 'H2':


        #on se trouve alors dans la condition HYPER
        if current_file.startswith('02') is True:


           #variables pour la suite

            DM_name = 'DM_HYPER_' + subj_name + '.csv'
            run = 'HYPER'

            #///////////mouvement regressors -AJUSTEMENTS SELON CONDITION\\\\\\\\\\

            #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
            #puisqu'on se trouve dans HYPER

            from function_split_reg_mvmnt_hyper import split_reg_hyper

            df_mvmnt_reg_HYPER = split_reg_hyper(df_mvmnt_reg_full, len(subj_data))



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


            from function_split_reg_mvmnt_hypo import split_reg_hypo

            df_mvmnt_reg_HYPO = split_reg_hypo(df_mvmnt_reg_full, len(subj_data))

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

            #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
            #puisqu'on se trouve dans HYPER


            from function_split_reg_mvmnt_hypo import split_reg_hypo

            df_mvmnt_reg_HYPO = split_reg_hypo(df_mvmnt_reg_full, len(subj_data))

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

            #On souhaite extraire la deuxième moitié de la matrice avec les régresseurs de mouvement
            #puisqu'on se trouve dans HYPER

            from function_split_reg_mvmnt_hyper import split_reg_hyper

            df_mvmnt_reg_HYPER = split_reg_hyper(df_mvmnt_reg_full, len(subj_data))


            #timestamps HYPER pour les sujets de H1
            timestamps =  os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx')

            ##/////////////////////////////////
            return timestamps, DM_name, df_mvmnt_reg_HYPER, run

    else:

        print('le fichier en cours n\'est pas un sujet appartenant à H1 ou H2')






