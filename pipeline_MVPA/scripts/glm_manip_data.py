

import numpy as np
import os
import pandas as pd
from os.path import exists
import scipy.io

def get_subj_data(data_dir, prefix = None):

    """
    -------------Fonction description----------

    Function that takes a directory of file and runs through all the file in it whilst it only list the path of the file starting
    with the prefix given as argument

    Example : in a directory you have many nii files, but you only want to use a sub-sample of them, e.g. the ones starting with the prefix swaf*

    -----------------Variables-----------------
    data_dir: path to a directory containing all files
    prefix : prefix of the file of interest that you want to list the path
    if prefix = None, all the files of the dir will be put in the list

    --------------------------------------------
    """


    #if prefix = None, all the files of the dir will be put in the list
    if prefix == None:

        #Extraction des volumes pour un sujet dans une liste
        ls_volumes_all = os.listdir(data_dir)

        #Crée une liste avec seulement les fichiers commencant par sw
        all_list = [x for x in ls_volumes_all]
        #print(swaf_list)

        #joindre le path avec les noms des volumes dans une liste
        #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
        subject_data = [os.path.join(data_dir, name) for name in all_list]


        return subject_data

    if prefix != None:
        #Extraction des volumes pour un sujet dans une liste
        ls_volumes_all = os.listdir(data_dir)

        #Crée une liste avec seulement les fichiers commencant par sw
        swaf_list = [x for x in ls_volumes_all if x.startswith(prefix)]
        #print(swaf_list)

        #joindre le path avec les noms des volumes dans une liste
        #--> on se retrouve avec une liste contenant les path de tous nos volumes d'intérêt
        subject_data = [os.path.join(data_dir, name) for name in swaf_list]


        return subject_data




def get_timestamps(data_path, subj_name, timestamps_path_root, return_df=None):

    """
    Parameters
    ----------

    data_path : path to fmri data in order to know in which conditions the subject is
    timestamps_path_root : root path to the timestamps
    subj_name : subject's name, e.g. APM_02_H2 to identify the particular cubjects (with different timestamps)
    return_df : if True, the function returns a pandas dataframe with the timestamps. If None, the path to timestamps will be returned

    Returns
    -------
    timestamps_path : a path to the timestamps file or if return_df =True, a pandas dataFrame which is named df_timestamps
    """


    #Read the file

#======================================

    #condition file is e.g. 02-Analgesia, it's the file that contains the fMRI volumes
    condition_file = os.path.basename(os.path.normpath(data_path))

    if 'Hyperalgesia' in condition_file: #need to return the right timestamps

        #TIMESTAMPS
        if subj_name == 'APM_02_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_HYPER.xlsx')#csv option

        elif subj_name == 'APM_05_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_HYPER.xlsx')

        elif subj_name == 'APM_17_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_HYPER.xlsx')

        elif subj_name == 'APM_20_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.mat'),simplify_cells =True)#.mat option
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_HYPER.xlsx')

            #timestamps HYPER pour les sujets normaux dans H2
        else :#For all other subjects
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_HYPER.mat'),simplify_cells =True)#.mat option
            else :
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_HYPER.xlsx')

        #if we are in the Analgesia/hypoalgesia condition
    elif 'Analgesia' in condition_file:

        if subj_name == 'APM_02_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM02_ANA.xlsx')

        elif subj_name == 'APM_05_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM05_ANA.xlsx')

        elif subj_name == 'APM_17_H2':
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM17_ANA.xlsx')

        elif subj_name == 'APM_20_H2':
            if return_df:
                timestamps =scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_APM20_ANA.xlsx')

        #timestamps HYPO/ANA for other 'normal' subjects
        else :
            if return_df:
                timestamps = scipy.io.loadmat(os.path.join(timestamps_path_root,r'ASTREFF_Model6_TxT_model3_multicon_ANA.mat'),simplify_cells =True)
            else:
                timestamps_path = os.path.join(timestamps_path_root, r'ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx')

    #-----return------
    #if the return is supposed to be a dataframe
    if return_df :

        df_timestamps = pd.concat([pd.DataFrame(timestamps['onsets']),pd.DataFrame(timestamps['durations']),pd.DataFrame(timestamps['names'])], axis = 1)
        df_timestamps.columns = ['onset', 'duration','trial_type']

        return df_timestamps
    #else return the path
    else:
        return timestamps_path



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


    #inverse_lenght = len(matrix_to_split) - target_lenght

    split_matrix = matrix_to_split.iloc[- target_lenght :]

    return split_matrix

def if_str_in_file(condition_file,subj_name):


    str_analgesia ='Analgesia'
    str_hyper = 'Hyperalgesia'
#defining design matrix name and mouvement regessors according to condition
    if str_analgesia in condition_file:
        condition = 'HYPO'
        DM_name = 'DM_HYPO_' + subj_name + '.csv' #Initializing the name under which the design matrix will be saved

    else:
        condition = 'HYPER'
        DM_name = 'DM_HYPER_' + subj_name + '.csv'

    return condition,DM_name

def exctract_files(path, str_t = None):
    for file in [i for i in os.listdir(path) if type in i or str_hyper in i ]:
