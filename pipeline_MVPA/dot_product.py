

def dot(path_to_img, path_output, signature = 'nps', conditions = None, resample_to_mask=True): 
    """
    Parameters
    ----------
    path_to_img : string
        path to the files (nii, hdr or img) on which the signature will be applied
    path_output : string 
        path to save the ouput of the function (i.e., dot product and related file)
    signature : string, default = 'nps'
        signature to apply on the data. The paths to the signature files are defined inside the function
    conditions : string, default = None
        name of the experimental condition

    Returns
    -------
    dot_array : numpy array
    subj_array : list
    """

    #----------Imports-------------
    import os
    import numpy as np
    import nibabel as nib
    import pandas as pd

    #---------Define signature path-----

    if signature == "nps":
        mask_path = "weights_NSF_grouppred_cvpcr.hdr"
    if signature == "siips":
        mask_path = "nonnoc_v11_4_137subjmap_weighted_mean.nii"
    if signature == "vps":
        mask_path = "bmrk4_VPS_unthresholded.nii"

    if resample_to_mask:
        resamp = "img_to_mask"
    if resample_to_mask == False:
        resamp = "mask_to_img"

    #----------Formatting files--------

    from A_data_prep import function_code as A

    #returns a list with all the paths of images in the provided argument 'path_to_img'
    data_path = A.get_subj_data(path_to_img)
    data_path = list(filter(lambda x: "hdr" in x, data_path))
    data_path.sort(reverse=True)
    data_path.sort()

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
            if resample_to_mask:
                resampled = image.resample_to_img(maps,mask)
            else:
                resampled = image.resample_to_img(mask,maps)

            print('image has been resample to shape : {} '.format(resampled.shape))


        #---------fitting images to 1 dimension------------
        #making mask and image a vector in order to dot product

        from nilearn.input_data import NiftiMasker
        #Fitting the masker of the 'mask' for which we want to dot product the beta maps
        masker_all = NiftiMasker(mask_strategy="whole-brain-template")

        #masker of the initial mask provided, in our case the mask is called NPS
        if resample_to_mask:
            masker_NPS = masker_all.fit_transform(mask)
        else:
            masker_NPS = masker_all.fit_transform(img)

        #fitting temporary masker for ongoing beta map :'maps'
        masker_tmp = masker_all.fit_transform(resampled)

        #---------Dot product---------

        #dot product of the image's masker with the mask(NPS)'s masker
        dot_res = np.dot(masker_tmp,masker_NPS.T)

        #storing the result in array
        dot_array = np.append(dot_array,dot_res)

        print('Computing dot product')
        print('=====================')

    df_res = pd.concat([pd.DataFrame(dot_array.T, columns = [f'dot_results_{resamp}_{signature}_{conditions}']),pd.DataFrame(subj_array, columns = ['files'])], axis=1)
    df_res.to_csv(os.path.join(path_output,f'results_dot_{resamp}_{signature}_{conditions}.csv'))

    return dot_array,subj_array


##Arguments of the function

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--signature", type=str, choices=['nps','siips','vps']) #signature on which to compute the dot product
    parser.add_argument("--path_to_data", type=str) #path to beta maps
    parser.add_argument("--path_output", type=str) #path to save the output
    parser.add_argument("--condition", type=str, default=None) #specify the experimental condition
    parser.add_argument("--resample_to_mask", type=bool, default=True) #how to resample the data. If true signature is resampled to data, otherwise data is resampled to signature
    args = parser.parse_args()

    dot_array, subj_array = dot(args.path_to_data, args.path_output, args.signature, args.condition, args.resample_to_mask)
