import numpy as np
import os
import glob
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from scripts import mvpa_prepping_data as prepping_data
from scripts import mvpa_building_model as building_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, permutation_test_score


def main_svc(data_input, save_path, kfold = 5, n_components_pca = 0.90, sub_data = False, which_train_data = False, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER'], cov_corr = True, binary = False):
    """
    This function serves to run a linear SVC on fmri data.
    arguments
    ---------

    """
    path, dirs, files = next(os.walk(data_input))
    #want to return a list of all the nii file from different folders
    print(path, dirs, files)
    data =[]
    gr = []
    group_indx = 1
    for dir in dirs:
        tmp_folder = glob.glob(os.path.join(path,dir, 'beta*'))
        for item in tmp_folder:
            data.append(item)
            gr.append(group_indx)#gr reflects which path or trial is associated with each participant (1 to n participants)
        group_indx += 1
    print(type(data))

    if sub_data != False: # For case where we need to use only a sub-part of data
        filt_data = []
        new_gr = []
        idx = 0
        for img in data:
            res = [ele for ele in sub_data if (ele in img)]
            if res:
            # if res if not empty, meaning that img path contain an element in sub_data, e.g. 'ANA' or 'N_ANA'
                filt_data.append(img)
                new_gr.append(gr[idx])
            idx += 1
        data.clear()
        gr.clear()
        data = filt_data
        gr = new_gr
        print(data)

    if binary or which_train_data != False: # controls for the case where binary = True, and which_train_data = ['', ''] so you don't want to encode Y as binary
        #df_target = prepping_data.encode_bin_classes(data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col
        df_target = prepping_data.encode_runs_as_classes(data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col   
        binary = True # if which_train_data is not False but binary = False to make sure model is binary
    else:
        df_target = prepping_data.encode_classes(data, gr)
    # Y data
    Y = np.array(df_target['target'])

    # masker
    masker, extract_X = prepping_data.extract_signal(data, mask = 'whole-brain-template', standardize = True) # extract_X is a (N obs. x N. voxels) structure. It will serve as X
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    check = np.isnan(X)
    print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X.shape)
    print(X)

    # PCA
    if n_components_pca < 1:
        pca = PCA(n_components = n_components_pca)
        #X = pca.fit_transform(X)

    # Split
    if which_train_data != False:

        X_train, X_test, Y_train, Y_test, y_train_gr_idx = prepping_data.train_test_iso_split(data,X,Y, which_train_data)
        split_gr = [gr[ele] for ele in y_train_gr_idx] #gr[:len(Y_train)] # split the group vector according to the split applied to X and Y
        binary = True
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=30)
        split_gr = gr[:len(Y_train)]

    print(type(X_train), type(X_test), type(Y_train), type(Y_test))
    print(X_train, X_test, Y_train, Y_test)
    print(split_gr)
    print('X_train.shape : {}, Y_train.shape : {}, X_test.shape : {}, Y_test.shape : {}'.format(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape))

    #K_FOLD MODELS
    if kfold > 0:
        dict_fold_results = building_model.train_test_models(X_train,Y_train, split_gr, kfold, binary = binary)


    # FINAL MODEL
    print('--Fitting final model--')
    model_clf = SVC(kernel="linear", probability = True, decision_function_shape = 'ovo')
    final_model = model_clf.fit(pca.fit_transform(X_train), list(Y_train))
    X_test = pca.transform(X_test)
    # Metrics
    final_results = dict()
    Y_pred = final_model.predict(X_test)

    final_row_metrics, cm, cr = building_model.compute_metrics_y_pred(Y_test, Y_pred) # cm : confusion matrix and cr : classification report
    if binary:
        final_y_score = final_model.predict_proba(X_test)[:, 1] # Here we only  take the second row of the output
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score))
    else:
        final_y_score =  final_model.predict_proba(X_test)
        final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score), multi_class = 'ovo')
    print('final_roc_auc_ovo: {}  :'.format(final_roc_auc_ovo))
    final_decision_func = final_model.decision_function(pca.transform(X))

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision']
    df_ypred_metrics = pd.DataFrame(columns = metrics_colnames)
    df_ypred_metrics.loc[0] = final_row_metrics

    dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func)

    # covariance matrix of X_test,Y_test
    wide_Y_test = building_model.hot_split_Y_test(Y_test,len(classes))
    if cov_corr:
        cov_x = np.cov(X_test.transpose().astype(np.float64))
        cov_y = np.cov(Y_test.transpose().astype(np.float64))
        print(cov_x.shape)
        print(cov_y.shape)
        #cov_mat = np.cov(X_test.transpose().astype(float),wide_Y_test.transpose().astype(float), rowvar = False, dtype = np.float64)
        print('cov_x and y shape')
        print(cov_x.shape)
        print(cov_y.shape)
        print(cov_y)
    # saving coeff., dict_final_results, final_model and fold_results
    #os.chdir(save_path)
    contrast_counter = 1
    for weights in final_model.coef_: # W is the weight vector

        (masker.inverse_transform(pca.inverse_transform(weights))).to_filename(f"coeffs_whole_brain_{contrast_counter}.nii.gz")
        print('weights shape')
        print(weights.shape)
        if cov_corr:
            # correction from Eqn 6 (Haufe et al., 2014)
            A = np.matmul(cov_x, weights)*(1/cov_y) # j'ai enlevé weights.transpose()
            print('A.shape : ')
            print(A.shape)
            print(masker.inverse_transform(pca.inverse_transform(A)).shape)
	    #print(masker.inverse_transform((pca.inverse_transform(A))).shape)
            # reproject to nii
            (masker.inverse_transform(pca.inverse_transform(A))).to_filename(f"eq6_adj_coeff_whole_brain_{contrast_counter}.nii.gz")

        contrast_counter += 1


    with open('final_results.pickle', 'wb') as handle:
        pickle.dump(dict_final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('final_results.pickle', 'rb') as handle:
        b = pickle.load(handle)

    filename_model = "final_model_SVC.pickle"
    pickle_out = open(filename_model,"wb")
    pickle.dump(final_model, pickle_out)
    pickle_out.close()

    if kfold > 0:
        with open('kfold_results.pickle', 'wb') as handle:
            pickle.dump(dict_fold_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('kfold_results.pickle', 'rb') as handle:
            b = pickle.load(handle)

    np.savez_compressed('XY_data_split.npz', X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test)
    #np.savez_compressed('cov_matrix.npz', cov_mat=cov_mat)
    if sub_data != False:
        print(f"model computed was acheived with the following subdata : {sub_data}")

#data_input = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\GLM_each_shock_4sub'
#save_out = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_decoding\test_4sub\test_final'


data_input = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/glm/each_shocks'
#  data_input = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\test_each_shock'

save_out = r'/home/p1226014/projects/def-rainvilp/p1226014/pain_decoding/results/mvpa '

main_svc(data_input, save_out, kfold = 5,n_components_pca = .90, which_train_data = False, sub_data = ['HYPER', 'ANA'], binary = True)  ###




























    


























































    














