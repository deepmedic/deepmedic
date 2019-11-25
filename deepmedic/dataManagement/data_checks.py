from __future__ import print_function

import glob
import os
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti


def text_to_html(_str):
    return '<pre><p>' + _str.replace('\n', '</p><p>') + '</p></pre>'


def get_html_colour(_str, colour='black', html=True):
    if html:
        return '<font color=\"' + colour + '\">' + _str + '</font>'
    else:
        return _str


def add_to_count_dict(key, _dict):
    if key in _dict:
        _dict[key] += 1
    else:
        _dict[key] = 1
    return _dict


def get_nifti_reader(filename):
    reader = sitk.ImageFileReader()

    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()

    return reader


def print_dict(item, prefix=''):
    ret = ''
    for key, value in item.items():
        ret += "{2}{0:10d}: {1}\n".format(value, key, prefix)
    return ret


class ResampleParams(object):
    def __init__(self, ref=None, origin=None, spacing=None, direction=None, size=None,
                 standard=False, suffix='', save_folder=None, thumbnails_folder=None, image_extension=None,
                 params=None):
        if params:
            self.ref = params.ref
            self.origin = params.origin
            self.spacing = params.spacing
            self.direction = params.direction
            self.size = params.size
            self.standard = params.standard
            self.suffix = params.suffix
            self.save_folder = params.save_folder
            self.thumbnails_folder = params.thumbnails_folder
            self.image_extension = params.image_extension
        else:
            self.ref = ref
            self.origin = origin
            self.spacing = spacing
            self.direction = direction
            self.size = size
            self.standard = standard
            self.suffix = suffix
            self.save_folder = save_folder
            self.thumbnails_folder = thumbnails_folder
            self.image_extension = image_extension


def get_image_dims_stats(image_list, do_pixs=False, do_dims=False, do_dtypes=False, do_direction=False,
                         open_image=False, disable_tqdm=False, tqdm_text='Getting Pixel Dimension Stats',
                         progress=None):
    if not (do_dims or do_pixs or do_dtypes or do_direction):
        return {}, {}, {}, {}
    dims_count = {}
    pix_dims_count = {}
    dtypes_count = {}
    direction_count = 0
    for image_path in tqdm(image_list, desc=tqdm_text, disable=disable_tqdm):
        image = NiftiImage(image_path)
        if open_image:
            image.open()

        if do_dims:
            dims_count = add_to_count_dict(image.get_size(), dims_count)

        if do_pixs:
            spacing = image.get_spacing()
            pix_dims_count = add_to_count_dict(spacing, pix_dims_count)

        if do_dtypes:
            dtypes_count = add_to_count_dict(image.get_pixel_type_string(), dtypes_count)

        if do_direction:
            if not image.is_in_std_radiology_view():
                direction_count += 1

        if progress is not None:
            progress.increase_value()

    return dims_count, pix_dims_count, dtypes_count, direction_count


def pix_check(pix_count, verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 1)
    ret = ''
    if len(pix_count) > 1:
        ret += get_html_colour('[FAILED]', 'red', html) + ' Pixel spacing check\n'
        if verbose:
            ret += prefix + 'Pixel dimensions do not match in between images\n'
            ret += prefix + 'We recommend resampling every image to isotropic pixel spacing (e.g. (1, 1, 1))\n'
            ret += prefix + 'pixel spacing Count:\n'
            ret += print_dict(pix_count, prefix)
    else:
        pix_dims = list(pix_count.keys())[0]
        for pix_dim in pix_dims:
            if not pix_dim == pix_dims[0]:
                ret += get_html_colour('[FAILED]', 'red', html) + ' Pixel spacing check\n'
                if verbose:
                    ret += prefix + 'Pixel dimensions do not match across dimensions\n'
                    ret += prefix + 'We recommend resampling every image to isotropic pixel spacing (e.g. (1, 1, 1))\n'
                    ret += prefix + 'Pixel spacing Count:\n'
                    ret += print_dict(pix_count, prefix)
                return

        ret += get_html_colour('[PASSED]', 'green') + ' Pixel dimensions check\n'
        if verbose:
            ret += prefix + 'Pixel Dimensions: ' + str(pix_dims)

    if html:
        ret = text_to_html(ret)

    return ret


def dims_check(dims_count, verbose=True, html=False):
    ret = ''
    prefix = ' '*(len('[PASSED]') + 1)
    if len(dims_count) > 1:
        ret += get_html_colour('[FAILED]', 'red', html) + ' Image dimensions check\n'
        if verbose:
            ret += prefix + 'Pixel dimensions do not match in between images\n'
            ret += prefix + 'We recommend resampling every image to the same pixel dimensions ' \
                            'for every dimension (e.g. (1, 1, 1))\n'
            ret += prefix + 'pixel Sizes Count:\n'
            ret += print_dict(dims_count, prefix)
    else:
        ret += get_html_colour('[PASSED]', 'green') + ' Image dimensions check\n'
        if verbose:
            ret += prefix + 'Image Dimensions: ' + str(list(dims_count.keys())[0])

    if html:
        ret = text_to_html(ret)

    return ret


def dtype_check(dtype_count, dtype=sitk.GetPixelIDValueAsString(sitk.sitkFloat32), verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 1)
    ret = ''
    if len(dtype_count) > 1:
        ret += get_html_colour('[FAILED]', 'red', html) + ' Data Type check\n'
        if verbose:
            ret += prefix + 'More than one data type\n'
            ret += prefix + 'We recommend resampling every image to ' + dtype + '\n'
            ret += prefix + 'Data Types Count:\n'
            ret += print_dict(dtype_count, prefix)
    else:
        if list(dtype_count.keys())[0] == dtype:
            ret += get_html_colour('[PASSED]', 'green') + ' Data Type check\n'
            if verbose:
                ret += prefix + 'Data Type: ' + str(list(dtype_count.keys())[0]) + '\n'
        else:
            ret += get_html_colour('[FAILED]', 'red', html) + ' Data Type check\n'
            ret += prefix + 'Sub-optimal data type. You might be using more memory than required storing your data ' \
                            'and subsequently increasing the loading time.\n'
            ret += prefix + 'We recommend resampling every image to ' + dtype + '\n'

    if html:
        ret = text_to_html(ret)

    return ret


def dir_check(dir_count, verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 1)
    ret = ''
    if dir_count:
        ret += get_html_colour('[FAILED]', 'red', html) + ' Orientation check\n'
        if verbose:
            ret += prefix + str(dir_count) + ' images are not in Standard Radiology View\n'
            ret += prefix + 'We recommend reorienting the images\n'
    else:
        ret += get_html_colour('[PASSED]', 'green') + ' Orientation check\n'
        ret += prefix + 'All images are in standard radiology view\n'

    if html:
        ret = text_to_html(ret)

    return ret


def run_checks(filelist, csv=False, pixs=False, dims=False, dtypes=False, dirs=False,
               disable_tqdm=False, html=False, progress=None):
    if csv:
        df = pd.read_csv(filelist)
        filelist = df['image']

    if progress is not None:
        progress.bar.setMaximum(len(filelist))

    (dims_count,
     spacing_count,
     dtype_count,
     direction_count) = get_image_dims_stats(filelist, do_dims=dims, do_pixs=pixs, do_dtypes=dtypes,
                                             disable_tqdm=disable_tqdm,
                                             tqdm_text='Running Image and Pixel Dimension Checks',
                                             progress=progress)
    ret = ''
    if dims:
        ret += dims_check(dims_count, html=html)
    if pixs:
        ret += pix_check(spacing_count, html=html)
    if dtypes:
        ret += dtype_check(dtype_count, html=html)
    if dirs:
        ret += dir_check(direction_count, html=html)

    return ret


def save_thumbnails(filelist, save_folder):
    for image_path in tqdm(filelist):
        image = NiftiImage(image_path)
        thumbnail_file = os.path.join(save_folder, image_path.split('/')[-1].split('.')[0])
        image.save_thumbnail(os.path.join(thumbnail_file + '.png'))


def resample_image_list(filelist, ref=None, origin=None, spacing=None, direction=None, size=None,
                        standard=False, suffix='', save_folder=None, thumbnails_folder=None, image_extension=None,
                        orientation=False, params=None):
    if params:
        ref = params.ref
        origin = params.origin
        spacing = params.spacing
        direction = params.direction
        size = params.size
        standard = params.standard
        suffix = params.suffix
        save_folder = params.save_folder
        thumbnails_folder = params.thumbnails_folder
        image_extension = params.image_extension
    if ref:
        ref_image = NiftiImage(ref)
    else:
        ref_image = None

    if save_folder:
        os.makedirs(save_folder)

    for image_path in filelist:
        path_split = image_path.split('.')
        image_name = path_split[0]
        if save_folder:
            image_save_name = os.path.join(save_folder, image_name.split('/')[-1])
        else:
            image_save_name = image_name
        if image_extension is None:
            image_extension = '.' + '.'.join(path_split[1:])
        if not suffix == '':
            suffix = '_' + suffix
        image = NiftiImage(image_path)
        if orientation:
            image.reorient()
        image.resample(ref_image=ref_image, origin=origin, spacing=spacing, direction=direction, size=size,
                       standard=standard)
        if save_folder:
            save_nifti(image, image_save_name + suffix + image_extension)
        if thumbnails_folder:
            thumbnail_file = os.path.join(thumbnails_folder, image_name.split('/')[-1])
            image.save_thumbnail(thumbnail_file + '.png')


def resize_images(image_list, masks, new_size, save_path, tqdm_text='Resizing images', disable_tqdm=False):
    for image_path, mask_path in tqdm(zip(image_list, masks), desc=tqdm_text, disable=disable_tqdm,
                                      total=len(image_list)):
        image = NiftiImage(image_path)
        mask = NiftiImage(mask_path)

        image.resize(new_size, mask, centre_mass=True, crop_mask=False)

        image.save(os.path.join(save_path, image_path.split('/')[-1]))
        # mask.save(os.path.join(save_path, mask_path.split('/')[-1]))


if __name__ == "__main__":
    # base_path = '/vol/vipdata/data/brain/adni/images/brain_adni2/'
    base_path = '/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2'
    # base_path = '/vol/biomedic2/bgmarque/deepmedic/examples/dataForExamples/brats2017_kostas2'
    # test_image = '/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2/Brats17TrainingData/HGG/Brats17_2013_2_1/Brats17_2013_2_1_t1.nii.gz'
    # img = NiftiImage(test_image)
    # for key in img.get_header_keys():
    #     print("{0}: {1}".format(key, img.reader.GetMetaData(key)))
    filelist = glob.glob(os.path.join(base_path, '**/*.nii.gz'), recursive=True)
    run_checks(filelist, dims=True, pixs=True, disable_tqdm=False)
    # dims_count, pixel_count = get_image_dims_stats(glob.glob(os.path.join(base_path, '**/*.nii.gz'), recursive=True), do_dims=False)
    # print('Dims Count')
    # print_dict(dims_count)
    # print('Pixel Count')
    # print_dict(pixel_count)
