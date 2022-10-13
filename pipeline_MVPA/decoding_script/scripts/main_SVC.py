import numpy as np
import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import prepping_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit, ShuffleSplit, permutation_test_score
import building_model

def main_ML(data_input):

    path, dirs, files = next(os.walk(filesInput))
    #want to return a list of all the nii file from different folders
    data =[]
    gr = []
    group_indx = 1
    for dir in dirs:
        tmp = glob.glob(os.path.join(path,dir, 'beta*'))
        print(tmp)
        for item in tmp:
            data.append(item)
            gr.append(group_indx)#gr reflects wich path or trial is associated with each participant (1 to n participants)
        group_indx += 1
    #----------------
    #Y data
    col_ls = ['filename', 'target', 'condition', 'group']
    df_target = pd.DataFrame(columns = col_ls)
    index = 0
    for file in data:

        #filename col
        filename = os.path.basename(os.path.normpath(file))#get filename
        df_target.loc[index, 'filename'] = filename #add file to index

        #digit and condition col
        print(filename, ': filename')
        if 'HYPO' in filename:
            if '_N_' in filename:
                target = 1 #hypo neutral
                cond = 'N_HYPO'

            else:#Hypo
                target = 2
                cond = 'HYPO'

        else : #hyper
            if 'N' in filename:
                target = 3
                cond = 'N_HYPER'
            else:
                target = 4
                cond = 'HYPER'
        #print('attributed : ', target, 'as target and :', cond, 'as condition')
        #print('-----------')
        df_target.loc[index, 'target'] = target #add to df
        df_target.loc[index, 'condition'] = cond

        index += 1
    df_target['group'] = gr
    Y = np.array(df_target['target'])#.reshape(y.shape[0:])
    #X
    #------masker------
    #extract_X is a 72 x 216 000 voxels structure. It will serve as X
    masker, extract_X = prepping_data.extract_signal(data)
    #---------------
    #split X Y
    #standardize X
    stand_X = StandardScaler().fit_transform(extract_X.T)
    X = stand_X.T
    check = np.isnan(X)
    print('check if NaN : ', np.isnan(np.min(X)), 'X SHAPE : ', X.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=42)
    #split the group vector according to the split of X and Y
    split_gr = gr[:len(Y_train)]
    print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)
    #---------------
    #K_FOLD MODELS
    y_pred = []
    model = []
    accuracy = []
    shuffle_method = GroupShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 33)
    model_clf = SVC(kernel="linear",class_weight='balanced')
    x_train, y_train, x_test, y_test, y_pred, model, accuracy, metrics = building_model.train_test_classify(X_train,Y_train , split_gr)

    #---------------
    #FINAL MODELS
    df_metrics = pd.DataFrame(columns=["accuracy", "precision"])
    print(df_metrics, ' df_metrics')
    model = SVC(kernel="linear",class_weight='balanced')
    final_model = model.fit(X_train,list(Y_train))
    Y_pred = final_model.predict(X_test)
    ###Scores### import building_model df_metrics =building_model.compute_metrics
    #  (Y_test, Y_pred, df_metrics, 'final', True)
    #accuracy = accuracy_score(list(Y_test), list(Y_pred))
    #print('accuracy ', df_metrics   )
    #SAVE
    contrast_counter = 1
    for element in final_model.coef_:

        (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{contrast_counter}.nii.gz")
        contrast_counter += 1
    df_metrics.to_csv('df_metrics.csv')
    np.save('Y_pred.npy', Y_pred)
    np.save('Y_test.npy', Y_test)
    #.save('accuracy_score.npy', np.array(accuracy))
    #for i in range(len(X_train)):
     #       filename = f"train_test_{i}.npz"
      #      np.savez(filename, X_train=X_train[i],Y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])


filesInput = r'/data/rainville/dylan_projet_ivado_decodage/results_GLM/each_shock'
#filesInput = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\GLM_each_shock'

main_ML(filesInput)



















    



























    


























































    














