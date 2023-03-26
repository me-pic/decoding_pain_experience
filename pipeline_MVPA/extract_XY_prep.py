import numpy as np
import os
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scripts import mvpa_prepping_data as prepping_data
from scripts import mvpa_building_model as building_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, permutation_test_score

'''Task list:
-régler problème avec df cond_target
-run extractXY on beluga
-faire sbatch file to run matlab script on beluga!
'''
def specif_svc(data_input, save_path,subj_folders = True, kfold = 5, n_components_pca = 0.90, sub_data = False, which_train_data = False, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'], cov_corr = True, binary = False, binary_fct = 'modulation'):
    """
    This function serves to run a linear SVC on fmri data.
    arguments
    ---------
    data_input : String; Path to fmri activation maps. It will extract the path of all imgs in path and its subfolders that has 'beta*' in name
    save_path : String; path to where results will be saved
    kfold : Int.; Number of folds to perform on the test set. If kfold = 0, kfold-cross-validation will be skipped. Default = 5
    n_components_pca : Float; Pourcentage of variance to keep in the principal component analysis for dimensianality reduction. If set to 0, no PCA will be applied. Default = 0.90
    sub_data :  Bool or string; By default, the model is built with all the data (sub_data = False), but list of string can be given to select only a sub-part of data to build model.
                E.g. sub_data = ['cond1', 'cond2'], only the data filename containing those strings will be selected to build the model (train and test). If = 'exception_ANAHYPER'
    which_train_data : Bool or string; By default, all the data is used to train/test the model (= False), but if a list of string is specified, the data having those strings in
                        their name will be selected to train the model and the rest of the data will be used to test the model. If which_train_data != False, binary = True automatically
                        E.g. which_train_data = ['cond1', 'cond2'], training is done with those conditions and the model is tested on 'cond3' and 'cond4'
                        *if which_train_data != False, be aware of the binary_fct
    **classes : String; Names of the conditions of the classes to classify. By default =  ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'].
    cov_corr : Bool; A covariance correction is applied on the final model's coefficients to reproject them in 3D brain space. The correction is from Haufe et al. (2014). Default = True
    binary : Bool; The model is initially built to be a multiclass model, but if binary = True, Y can be encoded bynarily according to binary_fct
    binary_fct : string; Only used in case binary = True. Different functions of class encoding are available. The default encoding function is based on manipulation vs neutral condition
                Default = 'modulation' Other choices = ['runs', ]
                Refer to scripts/mvpa_prepping.py data for functions' descritption
    """

    #extract data from path input
    data, gr, files = prepping_data.extract_data(data_input, extract_str = '*.hdr', folder_per_participant = subj_folders)
    if sub_data != False: # For case where we need to use only a sub-part of data
        data, gr, filename = prepping_data.keep_sub_data(data,gr, sub_data)

# Y data
    if binary or which_train_data != False: # controls for the case where binary = True, and which_train_data = ['', ''] so you don't want to encode Y as binary
        if binary_fct == 'modulation':
            df_target, cond_target  = prepping_data.encode_manip_classes(data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col

        elif binary_fct == 'runs':
            df_target, cond_target = encode_runs_as_classes(data, gr)
        binary = True # if which_train_data is not False but binary = False to make sure model is binary
    else: # regular multiclass case otherwise
        print('HERRRE')
        df_target = encode_classes(data, gr)
        print(df_target)
    Y = np.array(df_target['target'])

    # X data / vectorize fMRI activation maps
    masker, X = prepping_data.extract_signal(data, mask = 'whole-brain-template', standardize = True) # extract_X is a (N obs. x N. voxels) structure. It will serve as X
    #stand_X = StandardScaler().fit_transform(extract_X.T)
    #X = stand_X.T
    check = np.isnan(X)
    print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X.shape)

    # Ordering for groups
    XYgr = pd.concat([pd.DataFrame(X), pd.DataFrame({'Y': Y, 'gr' : gr})], axis=1)
    XYgr_ordered = XYgr.sort_values(by=XYgr.columns[-1]) #-1 being last col. e.i. 'gr'

    X = XYgr_ordered.iloc[:,:-2].to_numpy()
    Y = np.array(XYgr_ordered['Y'])
    gr = np.array(XYgr_ordered['gr'])

    os.chdir(save_out)
    np.savez_compressed('X_data.npz', X = X)
    np.savez_compressed('Y_data.npz', Y = Y)
    np.savez_compressed('gr_data.npz', gr = gr)

    #all
    #np.savez_compressed('X_data.npz', X = X, Y = Y, files = data, group = gr, df_target = df_target, X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)


#data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/each_shocks'
#data_input = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\result_GLM\test_txt_shock_3sub'
#save_out = r'E:\Users\Dylan\Desktop\UdeM_H22\E_PSY3008\result_GLM '

data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/contrast_singEvent_SPM'
save_out = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/matlab_svc'


#  which_train_data = ['oANA', 'oHYPER'] to train on hyper-hypo and test on rest




def transform_mat(path):
    from scipy.io import savemat
    import numpy as np
    import glob
    import os

    npzFiles = glob.glob(os.path.join(path,"*.npz"))
    print(npzFiles)
    for f in npzFiles:
        fm = os.path.splitext(f)[0]+'.mat'
        d = np.load(f,allow_pickle = True)
        savemat(fm, d)
        print('generated ', fm, 'from', f)



################################################3
def encode_classes(data, gr):

    #Y data
    y_colnames = ['filename', 'target', 'condition', 'group']
    df_target = pd.DataFrame(columns = y_colnames)
    index = 0
    for file in data:

            #filename col
        filename = os.path.basename(os.path.normpath(file))#get file name from path
        df_target.loc[index, 'filename'] = filename #add file to coord (index,'filnames')

        # encoding classes associated with each file in data
        if 'ANA' in filename:
            if 'N_ANA' in filename:
                target = -1 #hypo neutral
                cond = 'N_HYPO'

            else:#Hypo
                target = -1
                cond = 'HYPO'

        else : #hyper
            if 'N_HYPER' in filename:
                target = 1
                cond = 'N_HYPER'
            else:
                target = 1
                cond = 'HYPER'
            #print('attributed : ', target, 'as target and :', cond, 'as condition')
            #print('-----------')
        df_target.loc[index, 'target'] = target
        df_target.loc[index, 'condition'] = cond
        index += 1
    df_target['group'] = gr
    return df_target


def encode_runs_as_classes(data, gr):

    #Y data
    y_colnames = ['filename', 'target', 'condition', 'group']
    df_target = pd.DataFrame(columns = y_colnames)
    index = 0
    for file in data:

            #filename col
        filename = os.path.basename(os.path.normpath(file))#get file name from path
        df_target.loc[index, 'filename'] = filename #add file to coord (index,'filnames')

        # encoding classes associated with each file in data
        if 'ANA' in filename:
            if 'N_ANA' in filename:
                target = 1 #hypo neutral
                cond = 'Neutral'

            else:#Hypo
                target = 1
                cond = 'Modulation'

        else : #hyper
            if 'N_HYPER' in filename:
                target = 0
                cond = 'Neutral'
            else:
                target = -1
                cond = 'Modulation'
            #print('attributed : ', target, 'as target and :', cond, 'as condition')
            #print('-----------')
        df_target.loc[index, 'target'] = target
        df_target.loc[index, 'condition'] = cond

        index += 1
    df_target['group'] = gr
    cond_target = ['1 = HYPO_run', '2 = HYPER_run']

    return df_target, cond_target




specif_svc(data_input, save_out, subj_folders=False,n_components_pca = .90, sub_data = ['exception_ANAHYPER'], which_train_data = False, binary = False, binary_fct = 'runs')
transform_mat(save_out)









    



































