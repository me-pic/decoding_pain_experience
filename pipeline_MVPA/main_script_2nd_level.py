import numpy as np
import os
import pandas as pd
import nibabel as nib
import glob
from nilearn.glm.second_level import SecondLevelModel
import nibabel as nib
from nilearn import plotting


def run_second_level(data_input,save = None,condition = None, plot = None, img_format = '*.nii',participant_folder = True):

    """
    Parameters
    ----------
    data_input : string
        path to the files (nii, hdr or img) on which the signature will be applied
    save : string, default = None
        if not None, a path to where to save the output will be needed
    condition : string, default = None
        If save = True, a condition can be specified in the output file_name. e.g. condition = pain, the filename will be : beta_map_2nd_level_pain.nii
    plot : string, default = None
    img_format : string, default = '.nii'
        change according to the format of fmri images, e.g. '*.hdr' or '*.img'
    participant_folder : string, default = True
        If True, it's assumed that the path_to_img contains a folder for each participant that contain the fMRI image(s) that will be used to compute dot product.
        If False,it is assumed that all the images that will be used to compute dot product are directly in path_to_img

    Returns
    -------
    dot_array : numpy array
    subj_array : list
    """
    #---get data---
    if participant_folder:
        indiv_maps = glob.glob(os.path.join(data_input,'*',img_format))
    else:
       indiv_maps = glob.glob(os.path.join(data_input,img_format))

    #----second level design matrix----
    design_matrix = pd.DataFrame([1] * len(indiv_maps),
                                 columns=['intercept'])
    print(indiv_maps)
    #---second level model---
    second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
    second_level_model = second_level_model.fit(indiv_maps,
                                                design_matrix=design_matrix)
    beta_map_p = second_level_model.compute_contrast(output_type='p_value')
    z_map = second_level_model.compute_contrast(output_type='z_score')

    #===========EXEMPLE NILEARN=========

    #---THRESHOLD---
    from nilearn.glm import threshold_stats_img
    from nilearn import plotting

    #p < .001 uncorrected
    thresholded_map1, threshold1 = threshold_stats_img(
        z_map,
        alpha=.001,
        height_control='fpr',
        cluster_threshold=10000,
        two_sided=True,
    )
    print('p < .001 uncorrected threshold : ', threshold1)

    #fdr = .05
    thresholded_map2, threshold2 = threshold_stats_img(
        z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)

    #Bonferroni correction
    thresholded_map3, threshold3 = threshold_stats_img(
        z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)

    #-----PLOT-----
    #Raw results
    display = plotting.plot_stat_map(z_map, title='Raw z map')
    view = plotting.view_img_on_surf(z_map,colorbar = True,surf_mesh='fsaverage',title = 'Raw z map')
    #view.open_in_browser()

    #p < .001 uncorrected
    plotting.plot_stat_map(
    thresholded_map1, cut_coords=display.cut_coords, threshold=threshold1,
    title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    view = plotting.view_img_on_surf(thresholded_map1,threshold = threshold1,colorbar = True,surf_mesh='fsaverage',title = 'Thresholded z map, fpr <.001, clusters > 10000 voxels')
    view.open_in_browser()

    #fdr = .05
    plotting.plot_stat_map(thresholded_map2, cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fdr = .05',
                       threshold=threshold2)
    view = plotting.view_img_on_surf(thresholded_map2,threshold =threshold2 ,colorbar = True,surf_mesh='fsaverage',title = 'Thresholded z map, expected fdr = .05')
    view.open_in_browser()

    #Bonferroni correction
    plotting.plot_stat_map(thresholded_map3, cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fwer < .05',
                       threshold=threshold3)
    #view = plotting.view_img_on_surf(thresholded_map3,threshold =threshold3 ,colorbar = True,surf_mesh='fsaverage',title = 'Thresholded z map, expected fwer < .05')
    #view.open_in_browser()


    #plotting.show()
    """
    #==================================
    #---save---
    if save != None:
        if condition != None:
            beta_map_name = f'beta_map_2nd_level_{condition}.nii.gz'
        else:
            beta_map_name = 'beta_map_2nd_level.nii.gz'
        nib.save(beta_map, os.path.join(save, beta_map_name))

    #---plot---
    from nilearn import plotting
    if plot == True:
        from scipy.stats import norm
        p_val = 0.001
        p001_unc = norm.isf(p_val)
        display = plotting.plot_glass_brain(
            beta_map_p, threshold=p001_unc, colorbar=True, display_mode='z', plot_abs=False,
            title='second_level_p_all_shocks (unc p<0.001)')
        #plotting.show()

        display = plotting.plot_glass_brain(
            beta_map_z, threshold=p001_unc, colorbar=True, display_mode='z', plot_abs=False,
            title='second_level_z_all_shocks (unc p<0.001)')
        #plotting.show()
        #-----------
        #html_view = plotting.view_img(beta_map, threshold=0.0001, vmax=4, cut_coords=[-42, -16, 52], title='second level beta map')
        #html_view.open_in_browser()

        from nilearn import plotting

        view = plotting.view_img_on_surf(beta_map_z,colorbar = True,threshold='90%',surf_mesh='fsaverage',title = 'beta_map_z')

        view.open_in_browser()

        view = plotting.view_img_on_surf(beta_map_p,colorbar = True,threshold='90%',surf_mesh='fsaverage',title = 'beta_map_p')

        view.open_in_browser()

        #-----------------------
        title = ('Neutral shocks corrected p-values (FWER < 10%)')

        from nilearn.glm import threshold_stats_img
        _, threshold = threshold_stats_img(beta_map_z, alpha=.001, height_control='fdr')
        plotting.plot_stat_map(
        beta_map_z, threshold=threshold, colorbar=True,
        title='Neutral shocks z map ')
        plotting.show()

        html_view = plotting.view_img(beta_map_z, threshold=threshold, vmax=4, cut_coords=[-42, -16, 52],colorbar = True, title='second level beta map')
        html_view.open_in_browser()
        #-------
        import numpy as np
        from nilearn.image import get_data, math_img

        p_val = second_level_model.compute_contrast(output_type='p_value')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))"
                                .format(str(n_voxels)),
                                img=p_val)

        cut_coords = [0]
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there is
        # less than 10% probability to make a single false discovery (90% chance that
        # we make no false discovery at all).  This threshold is much more conservative
        # than the previous one.
        threshold = 1
        title = ('Group left-right button press: \n'
                 'parametric test (FWER < 10%)')
        display = plotting.plot_glass_brain(
            neg_log_pval, colorbar=True, display_mode='z', plot_abs=False, vmax=3,
            cut_coords=cut_coords, threshold=threshold, title=title)
        plotting.show()

"""
root_dir = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\GLM_all_shocks\all_shocks'
path_jeni = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\099_TxT_Individual_N-SHOCKS_files'
dir_to_save = r'C:\Users\Dylan\Desktop\UdeM_E22\Projet_Ivado_rainvillelab\results_GLM\GLM_second_level_all_shocks'

run_second_level(data_input=path_jeni,save = None, condition = None, plot = True,participant_folder=False, img_format = '*.hdr')
