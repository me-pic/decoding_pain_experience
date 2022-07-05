##fonction qui importe les données (volumes pour un participant)

import pandas as pd
import numpy as np
import os

                                         
#/////////////////////////////////////////////////////////
#root_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii'
#subject = 'APM_03_H1'

#variable utilisé comme paramètres de la fonction en guise de test
#data_dir = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\data_desmartaux\Nii\APM_03_H1\02-Analgesia'
#dir_anat = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\Analyses_conn\DATA_Desmateaux_Rainville\H1\APM_03_H1\01-MEMPRAGE\wms201301101300-0004-00001-000176-01.nii'
#timestamps_path = r'C:\Users\Dylan\Desktop\UdeM_H22\PSY3008\times_stamps\ASTREFF_Model6_TxT_model3_multicon_ANA.xlsx'

#DM_name = 'DM_HYPER_' + subject
#dir_to_save = os.path.join(root_dir,subject)

def create_DM(subject_data, timestamps_path, DM_name, regresseurs_mvmnt):

    #fonction qui compute une DM selon un timestamp, des régresseurs de mouvements,
    #un nom à donner à la DM et des volumes nifti
    import os
    import numpy as np
    import pandas as pd
    import nibabel as nb


    #Extraction des volumes pour un sujet
    from nilearn.image import concat_imgs, mean_img
    fmri_img = concat_imgs(subject_data)
    #print('SHAPE de FMRI IMG est {0}'.format(fmri_img.shape))

    #////////////TIMESTAMPS////////////////

    events = pd.read_excel(timestamps_path, header=None)
    #formatage des type des entrées et insertion de titres
    events = pd.DataFrame.transpose(events)
    events.rename(columns = {0:'onset', 1:'duration', 2:'trial_type'}, inplace = True)

    events['onset'] = events['onset'].astype(np.float64)
    events['duration'] = events['duration'].astype(np.float64)
    events['trial_type'] = events['trial_type'].astype(np.str)

    #///////////Design_Matrix/////////////////

    #paramètres de l'études
    tr = 3.  # repetition time, in seconds
    #slice_time_ref = 0.  # Sample at the beginning of each acquisition.
    #drift_model = 'Cosine'  # We use a discrete cosine transform to model signal drifts.
    #high_pass = .00233645  # The cutoff for the drift model is 0.01 Hz.
    #hrf_model = 'spm + derivative'  # The hemodynamic response function is the SPM canonical one.



    from nilearn.image import concat_imgs, mean_img
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.image import concat_imgs, mean_img


    #fmri_img = concat_imgs(subject_data)
    #mean_img = mean_img(fmri_img)
    n_scans = len(subject_data)

    frame_times = np.arange(n_scans) * tr


    design_matrices = []
    design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                hrf_model='spm',
                drift_model='cosine',
                high_pass=.00233645,
                add_regs = regresseurs_mvmnt) #array of shape(n_frames, n_add_reg)

    #/////////////PRINTS///////////////
    #print('SHAPE DE L\'IMAGE est {0}'.format(fmri_img.shape))
    print('SHAPE DU TIMESPTAMPS est {0}'.format(events.shape))

    #/////////////plot option///////////////

    #from nilearn.plotting import plot_design_matrix

    #plot_design_matrix(design_matrix)
    #import matplotlib.pyplot as plt
    #plt.show()
    #design_matrix.shape


    return design_matrix, fmri_img

