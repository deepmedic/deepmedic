from __future__ import print_function

import glob
import os
from tqdm import tqdm
import SimpleITK as sitk


def get_nifti_reader(filename):
    reader = sitk.ImageFileReader()

    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()

    return reader


def print_dict(item, prefix=''):
    for key, value in item.items():
        print("{2}{0:5d}: {1}".format(value, key, prefix))


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


def get_image_dims_stats(image_list, do_pixs=True, do_dims=False, tqdm_text='Getting Pixel Dimension Stats'):
    dims_count = {}
    pix_dims_count = {}
    for image_path in tqdm(image_list, desc=tqdm_text):
        image = NiftiImage(image_path)

        dims = image.get_image_dims()

        if dims in dims_count:
            dims_count[dims] += 1
        else:
            dims_count[dims] = 1

        dims = image.get_image_pixel_dims()

        if dims in pix_dims_count:
            pix_dims_count[dims] += 1
        else:
            pix_dims_count[dims] = 1

    print('\n')

    if do_pixs and do_dims:
        return dims_count, pix_dims_count
    elif do_pixs:
        return pix_dims_count
    else:
        return dims_count


def pix_check(filelist, verbose=True):
    pix_count = get_image_dims_stats(filelist, tqdm_text='Running Pixel Dimension Checks')
    prefix = ' '*(len('[PASSED]') + 1)
    if len(pix_count) > 1:
        print('[FAILED] Pixel dimensions check')
        if verbose:
            print(prefix + 'Pixel dimensions do not match in between images')
            print(prefix + 'We recommend resampling every image to the same pixel dimensions for every dimension '
                  '(e.g. (1, 1, 1))')
            print(prefix + 'pixel Sizes Count:')
            print_dict(pix_count, prefix)
    else:
        pix_dims = list(pix_count.keys())[0]
        for pix_dim in pix_dims:
            if not pix_dim == pix_dims[0]:
                print('[FAILED] Pixel dimensions check')
                if verbose:
                    print(prefix + 'Pixel dimensions do not match across dimensions')
                    print(prefix + 'We recommend resampling every image to the same pixel dimensions for every dimension '
                          '(e.g. (1, 1, 1))')
                    print(prefix + 'Pixel Sizes Count:')
                    print_dict(pix_count, prefix)
                return

        print('[PASSED] Pixel dimensions check')
        if verbose:
            print(prefix + 'Pixel Dimensions: ' + str(pix_dims))


def dims_check(filelist, verbose=True):
    dims_count = get_image_dims_stats(filelist, do_dims=True, do_pixs=False,
                                      tqdm_text='Running Image Dimension Checks')
    prefix = ' '*(len('[PASSED]') + 1)
    if len(dims_count) > 1:
        print('[FAILED] Images dimensions check')
        if verbose:
            print(prefix + 'Pixel dimensions do not match in between images')
            print(prefix + 'We recommend resampling every image to the same pixel dimensions for every dimension '
                  '(e.g. (1, 1, 1))')
            print(prefix + 'pixel Sizes Count:')
            print_dict(dims_count, prefix)
    else:
        print('[PASSED] Images dimensions check')
        if verbose:
            print(prefix + 'Image Dimensions: ' + str(list(dims_count.keys())[0]))


def run_checks(filelist):
    print('Running Checks')
    pix_check(filelist)



if __name__ == "__main__":
    # base_path = '/vol/vipdata/data/brain/adni/images/brain_adni2/'
    base_path = '/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2'
    # base_path = '/vol/biomedic2/bgmarque/deepmedic/examples/dataForExamples/brats2017_kostas2'
    # test_image = '/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2/Brats17TrainingData/HGG/Brats17_2013_2_1/Brats17_2013_2_1_t1.nii.gz'
    # img = NiftiImage(test_image)
    # for key in img.get_header_keys():
    #     print("{0}: {1}".format(key, img.reader.GetMetaData(key)))
    filelist = glob.glob(os.path.join(base_path, '**/*.nii.gz'), recursive=True)
    run_checks(filelist)
    # dims_count, pixel_count = get_image_dims_stats(glob.glob(os.path.join(base_path, '**/*.nii.gz'), recursive=True), do_dims=False)
    # print('Dims Count')
    # print_dict(dims_count)
    # print('Pixel Count')
    # print_dict(pixel_count)
