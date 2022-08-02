
filesInput = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\GLM_each_shock'
import numpy as np
import os
import glob
import pandas as pd

path, dirs, files = next(os.walk(filesInput))

#want to return a list of all the nii file from different folders

data =[]
gr = []
group_indx = 1

for dir in dirs:
    tmp = glob.glob(os.path.join(path,dir, 'beta*'))
    
    for item in tmp:
        data.append(item)
        gr.append(group_indx)#gr reflects wich path or trial is associated with each participant (1 to n participants)
    group_indx += 1



#----------------
#Y data

# to make the target

col_ls = ['filename', 'target', 'condition', 'group']
df_target = pd.DataFrame(columns = col_ls)

index = 0
for file in data:
    
    #filename col
    filename = os.path.basename(os.path.normpath(file))#get filename
    df_target.loc[index, 'filename'] = filename #add file to index 
   
    #digit and condition col
    #print(filename, ': filename')
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
    #test
    #print('attributed : ', target, 'as target and :', cond, 'as condition')
    #print('-----------')
    df_target.loc[index, 'target'] = target #add to df
    df_target.loc[index, 'condition'] = cond
    
    index += 1  

df_target['group'] = gr

Y = np.array(df_target['target'])#.reshape(y.shape[0:])

#----------------
#X

#------masker------
#X matrix
#extract_X is a 72 x 216 000 voxels structure. It will serve as X
from nilearn.masking import apply_mask
from nilearn.masker import NiftiMasker
from nilearn.image import resample_img
def extract_signal(data, mask="whole-brain-template", standardize = True):
    """
    Apply a mask to extract the signal from the data and save the mask
    in a html format

    Parameters
    ----------
    data: list of Niimg-like objects
    mask: strategy to compute the mask. By default the gray matter is extracted based on the MNI152 brain mask
    standardize: strategy to standardize the signal. The signal is z-scored by default

    Returns
    ----------
    masker_all: mask of the data
    masker_gm: array containing the extracted signal

    See also NifitMasker documentation
    """
    masker_all = NiftiMasker(mask_strategy = mask,standardize=standardize, verbose = 1, reports = True)
    
    masker_gm = masker_all.fit_transform(data)
    print("mean: ", round(masker_gm.mean(),2), "\nstd: ", round(masker_gm.std(),2))
    print("Signal size: ", masker_gm.shape)

    report = masker_all.generate_report()
    report.save_as_html("masker_report.html")

    return masker_all, masker_gm

masker, extract_X = extract_signal(data)

#---------------
#split X Y
from sklearn.preprocessing import StandardScaler

stand_X = StandardScaler().fit_transform(extract_X.T)
X = stand_X.T

check = np.isnan(X)
print('check if NaN : ', np.isnan(np.min(X)), 'X SHAPE : ', X.shape)

import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30, random_state=42)

#split the group vector according to the split of X and Y
split_gr = gr[:len(Y_train)]

print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)

#---------------
#K_FOLD MODELS
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit, ShuffleSplit, permutation_test_score
y_pred = []
model = []
accuracy = []

#kfold object
shuffle_method = GroupShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 33)
#model
model_clf = SVC(kernel="linear",class_weight='balanced')

x_train, y_train, x_test, y_test, y_pred, model, accuracy, metrics = building_model.train_test_classify(X_train,Y_train , split_gr)
                                                                                              

#---------------
#FINAL MODELS
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score

df_metrics = pd.DataFrame(columns=["r2", "mae", "mse", "rmse"])

model = SVC(kernel="linear",class_weight='balanced')
final_model = model.fit(X_train,list(Y_train))

Y_pred = final_model.predict(X_test)
print('y_pred : ', y_pred)

###Scores###
df_metrics = building_model.compute_metrics(y_test[i], y_pred[i], df_metrics, i, True)

accuracy = accuracy_score(list(Y_test), list(Y_pred))
print('accuracy', accuracy)

#SAVE
contrast_counter = 1
for element in final_model.coef_:
    
    (masker.inverse_transform(element)).to_filename(f"coefs_whole_brain_{contrast_counter}.nii.gz")
    contrast_counter += 1

df_metrics.to_csv('df_metrics.csv')

np.save('Y_pred.npy', Y_pred)
np.save('Y_test.npy', Y_test)
np.save('accuracy_score.npy', np.array(accuracy))


#for i in range(len(X_train)):
 #       filename = f"train_test_{i}.npz"
  #      np.savez(filename, X_train=X_train[i],Y_train=y_train[i],X_test=X_test[i],y_test=y_test[i],y_pred=y_pred[i])

