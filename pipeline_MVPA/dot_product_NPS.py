

#Manual tests
#mask_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\NPS_wager_lab\NPS_share\weights_NSF_grouppred_cvpcr.hdr'

#beta_map_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\beta_1st_level_all_shocks\beta_map_APM_05_H2_HYPER_all_shocks.nii'
#test_path = 'ceci est un test path pour commit sur une autre branche'




#put the two imgs into 2D arrays to facilitate dot product
def dot(path_to_img, mask_path):


#path_to_img : path to the folder of all the images
#mask : ath or nifti file of the mask with which the images will be dot product with

#return a csv file with one column as the results of the dot products and the second column with the name of
 #   the file/participant


    #----------Imports-------------
    import os
    import numpy as np
    import nibabel as nib


    #----------Formatting files--------

    from A_data_prep import function_code as A

    #returns a list with all the paths of images in the provided argument 'path_to_img'
    data_path = A.get_subj_data(path_to_img)


    #Load the mask as a nii file
    mask = nib.load(mask_path)
    print('mask shape :  {} '.format(mask.shape))
    #------------------------


    #Initializing empty arrays for results)
    dot_array = np.array([])
    subj_array = []
    #----------Main loop--------------

    #For each map/nii file in the provided path
    for maps in data_path:

        #load ongoing image
        img = nib.load(maps)

        #saving the file/subject's name in a list
        subj = os.path.basename(os.path.normpath(maps))
        subj_array.append(subj)

        #if image is not the same shape as the mask, the image will be resample
        #mask is the original mask to dot product with the provided images
        if img.shape != mask.shape:

        #---------Resampling--------
            print('Resampling : ' + os.path.basename(os.path.normpath(maps)))

            #resampling img to mask's shape
            from nilearn import image
            resampled_img = image.resample_to_img(maps,mask)

            print('image has been resample to shape : {} '.format(resampled_img.shape))


        #---------fitting images to 1 dimension------------
        #making mask and image a vector in order to dot product

        from nilearn.input_data import NiftiMasker
        #Fitting the masker of the 'mask' for which we want to dot product the beta maps
        masker_all = NiftiMasker(mask_strategy="whole-brain-template")

        #masker of the initial mask provided, in our case the mask is called NPS
        masker_NPS = masker_all.fit_transform(mask)

        #fitting temporary masker for ongoing beta map :'maps'
        masker_tmp = masker_all.fit_transform(resampled_img)

        #---------Dot product---------

        #dot product of the image's masker with the mask(NPS)'s masker
        dot_res = np.dot(masker_tmp,masker_NPS.T)

        #storing the result in array
        dot_array = np.append(dot_array,dot_res)

        print('Computing dot product')
        print('=====================')

    return dot_array,subj_array


    #--------Saving to csv---------

##Arguments of the function

NPS_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\NPS_wager_lab\NPS_share\weights_NSF_grouppred_cvpcr.img'
imgs_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\beta_1st_level_all_shocks'



dot_array, subj_array = dot(imgs_path,NPS_path)

print(dot_array, subj_array)
