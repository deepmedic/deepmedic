from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti
import os


def preprocess(image_path, output_dir='.', orientation_corr=True, resample_imgs=True, change_pixel_type=True,
               pixel_type='float32', thresh_low=None, thresh_high=None, mask_pixel_type='uint8',
               spacing=(1, 1, 1), create_mask=False, threshold=False, thresh_low_cut=None, thresh_high_cut=None,
               norm_range=False, low_range_orig=None, high_range_orig=None,
               low_range_target=None, high_range_target=None):

    image = NiftiImage(image_path, None, None, channel_names=['Channel_0'])

    # convert type
    if change_pixel_type:
        image.change_pixel_type(pixel_type)

    # reorient
    if orientation_corr:
        image.reorient()

    # resample (spacing)
    if resample_imgs:
        image.resample(spacing=spacing)

    # create  <--------------------------------------------------------- R E V I E W --------------------
    if create_mask:
        image.get_mask(thresh_low, thresh_high)
        if mask_pixel_type:
            image.mask.change_pixel_type(mask_pixel_type)

    # cutoff
    if threshold:
        image.thresh_cutoff(thresh_low_cut, thresh_high_cut)

    # normalise range
    if norm_range:
        image.norm_range(low_range_orig, high_range_orig, low_range_target, high_range_target)

    # save image
    if output_dir:
        save_nifti(image.channels['Channel_0'].open(), os.path.join(output_dir, 'preproc.nii.gz'))
        if image.mask:
            save_nifti(image.mask.open(), os.path.join(output_dir, 'mask.nii.gz'))
        if image.target:
            save_nifti(image.target.open(), os.path.join(output_dir, 'target.nii.gz'))
