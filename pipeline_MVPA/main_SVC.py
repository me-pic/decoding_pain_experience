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
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, permutation_test_score


def main_svc(data_input, save_path, kfold = 5, iso_data = False, classes = ['N_HYPO', 'HYPO', 'N_HYPER', 'HYPER']):

    path, dirs, files = next(os.walk(data_input))
        #want to return a list of all the nii file from different folders

    data =[]
    gr = []
    group_indx = 1
    for dir in dirs:
        tmp_folder = glob.glob(os.path.join(path,dir, 'beta*'))
        for item in tmp_folder:
            data.append(item)
            gr.append(group_indx)#gr reflects which path or trial is associated with each participant (1 to n participants)
        group_indx += 1
        #----------------
    # For the case scenario where you want to train with a subset of conditions and test on other conditions/classes
    if iso_data:
        filter_train_data = [file for file in data if 'HYPER' or 'N_HYPER' in file]
        print(filter_train_data)
        print(len(filter_train_data))
        filter_test_data = [file for file in data if 'ANA' or 'N_ANA' in file]
        print(filter_test_data)
        print(len(filter_test_data))
        df_target =  prepping_data.encode_classes(filter_train_data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col
    else:
        df_target =  prepping_data.encode_classes(data, gr) # df_target has ['filename', 'target', 'condition', 'group'] as col

        # Y data
        Y = np.array(df_target['target'])

    # masker
    if load_X:
        os.chdir(save_path)
        X = np.load('X.np')
        if iso_data:
            masker, extract_X = prepping_data.extract_signal(filter_train_data)
        else:
            masker, extract_X = prepping_data.extract_signal(data) # extract_X is a (N obs. x N. voxels) structure. It will serve as X

        stand_X = StandardScaler().fit_transform(extract_X.T)
        X = stand_X.T

    check = np.isnan(X)
    print('check if NaN : ', np.isnan(np.min(X)), '. X SHAPE : ', X.shape)

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=42)
    split_gr = gr[:len(Y_train)] # split the group vector according to the split applied to X and Y
    print('X_train.shape : {}, Y_train.shape : {}, X_test.shape : {}, Y_test.shape : {}'.format(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape))

    #K_FOLD MODELS
    if kfold > 0:
        dict_fold_results = building_model.train_test_models(X_train,Y_train, split_gr, kfold)

    # FINAL MODEL
    print('--Fitting final model--')
    model_clf = SVC(kernel="linear",class_weight='balanced', probability = True)
    final_model = model_clf.fit(X_train , list(Y_train))

    # Metrics
    final_results = dict()

    final_y_score =  final_model.predict_proba(X_test)
    Y_pred = final_model.predict(X_test)
    final_row_metrics, cm, cr = building_model.compute_metrics_y_pred(Y_test, Y_pred) # cm : confusion matrix and cr : classification report
    final_roc_auc_ovo = roc_auc_score(list(Y_test), np.array(final_y_score), multi_class = 'ovo')
    print('final_roc_auc_ovo: {}  :'.format(final_roc_auc_ovo))
    final_decision_func = final_model.decision_function(X)

    metrics_colnames = ['accuracy', 'balanced_accuracy', 'precision']
    df_ypred_metrics = pd.DataFrame(columns = metrics_colnames)
    df_ypred_metrics.loc[0] = final_row_metrics

    dict_final_results = dict(y_pred_metrics = df_ypred_metrics, Y_pred = Y_pred, Y_score = final_y_score, roc_auc_ovo = final_roc_auc_ovo, confusion_matrix = cm, classification_report = cr, decision_function = final_decision_func)
    
    # covariance matrix of X_test,Y_test
    wide_Y_test = building_model.hot_split_Y_test(Y_test,len(classes))
    cov_mat = np.cov(X_test.transpose().astype(float),wide_Y_test.transpose().astype(float), rowvar = False, dtype = np.float32)

    # saving coeff., dict_final_results, final_model and fold_results
    os.chdir(save_path)
    contrast_counter = 1
    for element in final_model.coef_:
        (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{contrast_counter}.nii.gz")
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
    np.savez_compressed('cov_matrix.npz', cov_mat=cov_mat)


#data_input = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_GLM\test_res_GLM\GLM_each_shock_4sub'
#save_out = r'C:\Users\Dylan\Desktop\UM_Bsc_neurocog\E22\Projet_Ivado_rainvillelab\results_decoding\test_4sub\test_final'

data_input = r'/data/rainville/dylan_projet_ivado_decodage/results_GLM/each_shocks'
save_out = r'/data/rainville/dylan_projet_ivado_decodage/results_mvpa/model2'

main_svc(data_input, save_out, kfold = 5, iso_data = False)



















    



























    


























































    














