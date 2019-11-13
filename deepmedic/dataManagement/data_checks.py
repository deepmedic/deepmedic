from __future__ import print_function

import glob
import os
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk


def text_to_html(_str):
    return '<pre><p>' + _str.replace('\n', '</p><p>') + '</p></pre>'


def get_html_colour(_str, colour='black', html=True):
    if html:
        return '<font color=\"' + colour + '\">' + _str + '</font>'
    else:
        return _str


def get_nifti_reader(filename):
    reader = sitk.ImageFileReader()

    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()

    return reader


def print_dict(item, prefix=''):
    ret = ''
    for key, value in item.items():
        ret += "{2}{0:5d}: {1}".format(value, key, prefix)
    return ret


class NiftiImage(object):

    def __init__(self, filename):
        self.reader = get_nifti_reader(filename)

    def get_num_dims(self):
        return int(self.reader.GetMetaData('dim[0]'))

    def get_image_dims(self, num_dims=3):
        if num_dims is None:
            num_dims = self.get_num_dims()
        dims = []
        for i in range(num_dims):
            dims.append(int(self.reader.GetMetaData('dim[' + str(i+1) + ']')))

        return tuple(dims)

    def get_image_pixel_dims(self, pix_dims=3):
        if pix_dims is None:
            pix_dims = self.get_num_dims()
        dims = []
        for i in range(pix_dims):
            dims.append(float(self.reader.GetMetaData('pixdim[' + str(i+1) + ']')))

        return tuple(dims)

    def get_header_keys(self):
        return self.reader.GetMetaDataKeys()


def get_image_dims_stats(image_list, do_pixs=True, do_dims=True, disable_tqdm=False,
                         tqdm_text='Getting Pixel Dimension Stats', progress=None):
    if not (do_dims or do_pixs):
        return {}, {}
    dims_count = {}
    pix_dims_count = {}
    for image_path in tqdm(image_list, desc=tqdm_text, disable=disable_tqdm):
        image = NiftiImage(image_path)

        if do_dims:
            dims = image.get_image_dims()

            if dims in dims_count:
                dims_count[dims] += 1
            else:
                dims_count[dims] = 1

        if do_pixs:
            dims = image.get_image_pixel_dims()

            if dims in pix_dims_count:
                pix_dims_count[dims] += 1
            else:
                pix_dims_count[dims] = 1

        if progress is not None:
            progress.increase_value()

    return dims_count, pix_dims_count


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


def run_checks(filelist, csv=False, pixs=False, dims=False, disable_tqdm=False, html=False, progress=None):
    if csv:
        df = pd.read_csv(filelist)
        filelist = df['image']

    if progress is not None:
        progress.bar.setMaximum(len(filelist))

    dims_count, scaling_count = get_image_dims_stats(filelist, do_dims=dims, do_pixs=pixs,
                                                     disable_tqdm=disable_tqdm,
                                                     tqdm_text='Running Image and Pixel Dimension Checks',
                                                     progress=progress)
    ret = ''
    if dims:
        ret += dims_check(dims_count, html=html)
    if pixs:
        ret += pix_check(scaling_count, html=html)

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
