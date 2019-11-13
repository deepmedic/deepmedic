from __future__ import print_function

import glob
import os
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

from deepmedic.dataManagement.nifti_image import NiftiImage


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


def get_image_dims_stats(image_list, do_pixs=False, do_dims=False, do_dtypes=False, do_direction=False,
                         open_image=False, disable_tqdm=False, tqdm_text='Getting Pixel Dimension Stats',
                         progress=None):
    if not (do_dims or do_pixs or do_dtypes or do_direction):
        return {}, {}, {}, {}
    dims_count = {}
    pix_dims_count = {}
    dtypes_count = {}
    direction_count = {}
    for image_path in tqdm(image_list, desc=tqdm_text, disable=disable_tqdm):
        image = NiftiImage(image_path)
        if open_image:
            image.open()

        if do_dims:
            dims_count = add_to_count_dict(image.get_size(), dims_count)

        if do_pixs:
            pixel_dims = image.get_spacing()
            pix_dims_count = add_to_count_dict(pixel_dims, pix_dims_count)

        if do_dtypes:
            dtypes_count = add_to_count_dict(image.get_pixel_type_string(), dtypes_count)

        if do_direction:
            direction = image.get_direction()
            direction_count = add_to_count_dict(direction, direction_count)

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


def run_checks(filelist, csv=False, pixs=False, dims=False, dtypes=False, disable_tqdm=False, html=False, progress=None):
    if csv:
        df = pd.read_csv(filelist)
        filelist = df['image']

    if progress is not None:
        progress.bar.setMaximum(len(filelist))

    (dims_count,
     scaling_count,
     dtype_count, _) = get_image_dims_stats(filelist, do_dims=dims, do_pixs=pixs, do_dtypes=dtypes,
                                            disable_tqdm=disable_tqdm,
                                            tqdm_text='Running Image and Pixel Dimension Checks',
                                            progress=progress)
    ret = ''
    if dims:
        ret += dims_check(dims_count, html=html)
    if pixs:
        ret += pix_check(scaling_count, html=html)
    if dtypes:
        ret += dtype_check(dtype_count, html=html)

    return ret


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
