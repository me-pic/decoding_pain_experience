


mask_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\NPS_wager_lab\NPS_share\weights_NSF_grouppred_cvpcr.hdr'

beta_map_path = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\beta_1st_level_all_shocks\beta_map_APM_05_H2_HYPER_all_shocks.nii'
test_path = 'ceci est un test path pour commit sur une autre branche'

#/////////////////////////////////////////
#////////////////////////////////////////
#resampling
import numpy as np
import nibabel as nib

mask = nib.load(mask_path)
print('mask shape :  {} '.format(mask.shape))

beta_map = nib.load(beta_map_path)
print('input shape :  {} '.format(beta_map.shape))

#======plotting checks======
"""
from nilearn import plotting
html_view = plotting.view_img(beta_map, threshold=2, vmax=4, cut_coords=[-42, -16, 52], title='beta map')

html_view.open_in_browser()

html_view = plotting.view_img(mask, threshold=2, vmax=4, cut_coords=[-42, -16, 52], title='NPS')

html_view.open_in_browser()
"""
#===========================
if mask.shape != beta_map.shape:
    print('will resample')

    data = np.array(beta_map.dataobj).astype("uint8")

    from nilearn import image

    resampled_mask = image.resample_to_img(mask,beta_map)

    print('Mask has been resample to shape : {} '.format(resampled_mask.shape))
    #print(type(resampled_mask))

    #========Dot product==========

    from nilearn.maskers import NiftiMasker

    #beta_mask = NiftiMasker(beta_map)
    masker_all = NiftiMasker(mask_strategy="whole-brain-template")
    masker_beta = masker_all.fit_transform(beta_map_path)
    masker_beta.shape


#put the two imgs into 2D arrays to facilitate dot product


