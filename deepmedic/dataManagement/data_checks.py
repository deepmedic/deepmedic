from __future__ import print_function

import glob
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import SimpleITK as sitk

from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti, inv_pixel_type_dict


def text_to_html(_str):
    return '<font size="2"><pre><p>' + _str.replace('\n', '</p><p>') + '</p></pre></font>'


def get_html_colour(_str, colour='black', html=True):
    if html:
        return '<font color=\"' + colour + '\">' + _str + '</font>'
    else:
        return _str


def get_bold_text(_str, html=True):
    if html:
        return '<b>' + _str + '</b>'
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


def print_dict(item, prefix='', max_num=3):
    ret = ''
    if len(item) <= max_num + 2:
        max_num += 2
    n = 0
    sorted_item = sorted(item.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for key, value in sorted_item:
        if n == max_num:
            (last_key, last_value) = sorted_item[-1]
            ret += "{0}{1}....\n".format(prefix, ' ' * int(11 + (len(str(last_value)) + (len(str(last_key)) / 2 - 2))))
            ret += "{2}{0:10d}: {1}\n".format(last_value, last_key, prefix)
            break
        else:
            ret += "{2}{0:10d}: {1}\n".format(value, key, prefix)
            n += 1
    return ret[:-1]


def print_list(item, prefix=''):
    ret = ''
    for value in sorted(item):
        ret += "{1}{0}\n".format(value, prefix)
    return ret[:-1]


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


def get_image_dims_stats(image_list, do_pixs=False, do_dims=False, do_dtypes=False, do_direction=False, do_size=False,
                         do_intra_dims=False, do_intra_size=False, open_image=False, col_names=None, id_col=None,
                         disable_tqdm=False, tqdm_text='Getting Pixel Dimension Stats', progress=None):
    if not (do_dims or do_pixs or do_dtypes or do_direction or do_size or do_intra_size):
        return [], [], {}, {}, {}, {}, {}
    intra_dims_count = []
    intra_size_count = []
    dims_count = {}
    pix_dims_count = {}
    dtypes_count = {}
    size_count = {}
    direction_count = 0
    if type(image_list) == list:
        image_list = pd.DataFrame(data={'Image': image_list})
    if col_names is None:
        col_names = [a for a in image_list.columns if a.startswith('Channel_') or a == 'Image']
    if id_col is None:
        if 'Id' not in image_list.columns:
            image_list['Id'] = list(range(len(image_list)))
        id_col = 'Id'

    for _, row in tqdm(image_list.iterrows(), desc=tqdm_text, disable=disable_tqdm):
        intra_dims_count_subj = {}
        intra_size_count_subj = {}
        for col in col_names:
            image_path = row[col]
            image = NiftiImage(image_path)

            if open_image:
                image.open()

            if do_dims or do_intra_dims:
                dims = image.get_size()
                if do_intra_dims:
                    intra_dims_count_subj = add_to_count_dict(dims, intra_dims_count_subj)

                if do_dims:
                    dims_count = add_to_count_dict(dims, dims_count)

            if do_pixs:
                spacing = image.get_spacing()
                spacing = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, spacing))
                pix_dims_count = add_to_count_dict(spacing, pix_dims_count)

            if do_dtypes:
                dtypes_count = add_to_count_dict(image.get_pixel_type_string(), dtypes_count)

            if do_size or do_intra_size:
                size = tuple(int(round(sp * dm)) for sp, dm in zip(image.get_spacing(), image.get_size()))
                if do_size:
                    size_count = add_to_count_dict(size, size_count)
                if do_intra_size:
                    intra_size_count_subj = add_to_count_dict(size, intra_size_count_subj)

            if do_direction:
                if not image.is_in_std_radiology_view():
                    direction_count += 1

        if progress is not None:
            progress.increase_value()

        if do_intra_size or do_intra_dims:
            id = row[id_col]
            if do_intra_size:
                if len(intra_size_count_subj) > 1:
                    intra_size_count += [id]
            if do_intra_dims:
                if len(intra_dims_count_subj) > 1:
                    intra_dims_count += [id]

    return intra_dims_count, intra_size_count, dims_count, pix_dims_count, dtypes_count, direction_count, size_count


def pix_check(pix_count, verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 2)
    ret = ''
    suggested = None
    passed = True
    if len(pix_count) > 1:
        ret += get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' Pixel Spacing Check\n')
        suggested = (1, 1, 1)
        if verbose:
            ret += prefix + 'Pixel dimensions do not match in between images\n'
            ret += prefix + 'Consider resampling to isotropic pixel spacing (e.g. (1, 1, 1))\n'
            ret += prefix + 'Note that resampling will change the image (pixel) dimensions\n'
            ret += prefix + 'Consider resizing in addition to resampling\n'
            ret += prefix + 'We recommend all images have the same spacing and dimensions.\n'
            ret += prefix + 'Pixel Spacing Count (mm):\n'
            ret += print_dict(pix_count, prefix)
    else:
        pix_dims = tuple(pix_count.keys())[0]
        for pix_dim in pix_dims:
            if not pix_dim == pix_dims[0]:
                ret += get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' Pixel Spacing Check\n')
                suggested = (1, 1, 1)
                if verbose:
                    ret += prefix + 'Pixel dimensions do not match across dimensions\n'
                    ret += prefix + 'Consider resampling to isotropic pixel spacing (e.g. (1, 1, 1))\n'
                    ret += prefix + 'Note that resampling will change the image (pixel) dimensions\n'
                    ret += prefix + 'Consider resizing in addition to resampling.\n'
                    ret += prefix + 'We recommend all images have the same spacing and dimensions.\n'
                    ret += prefix + 'Pixel spacing Count (mm):\n'
                    ret += print_dict(pix_count, prefix)
                passed = False

        if passed:
            ret += get_bold_text(get_html_colour(' [PASSED]', 'green') + ' Pixel Spacing Check\n')
            if verbose:
                ret += prefix + 'Pixel Spacing (mm): ' + str(pix_dims)

    if html:
        ret = text_to_html(ret)

    return ret, suggested


def get_max_dims(dims_count):
    max_size = np.zeros(max([len(i) for i in dims_count.keys()]))
    for size in dims_count.keys():
        for i in range(len(size)):
            if size[i] > max_size[i]:
                max_size[i] = size[i]
    return tuple(max_size)


def dims_check(dims_count, pixel=False, verbose=True, html=False):
    ret = ''
    prefix = ' '*(len('[PASSED]') + 2)
    if pixel:
        dim_type = 'Pixel'
        dim_units = ''
    else:
        dim_type = 'Actual'
        dim_units = ' (mm)'
    suggested = None
    if len(dims_count) > 1:
        ret += get_bold_text(get_html_colour('   [INFO]', 'salmon', html) + ' Image (' + dim_type + ') Dimensions Check\n')
        # suggested = get_max_dims(dims_count)
        if verbose:
            ret += prefix + 'Image dimensions do not match across images\n'
            ret += prefix + 'Some applications (e.g. classification) require uniformly\n'
            ret += prefix + 'sized images.\n'
            ret += prefix + 'Consider the needs of your application.\n'
            ret += prefix + 'Image Dimensions Count' + dim_units + ':\n'
            ret += print_dict(dims_count, prefix)

    else:
        ret += get_bold_text(get_html_colour('   [INFO]', 'mediumaquamarine') + ' Image (' + dim_type + ') Dimensions Check\n')
        if verbose:
            ret += prefix + 'All images have the same dimensions\n'
            ret += prefix + 'Image Dimensions: ' + str(list(dims_count.keys())[0])

    if html:
        ret = text_to_html(ret)

    return ret, suggested


def intra_dims_check(dims_count, pixel=False, cols=None, verbose=True, html=False):
    ret = ''
    prefix = ' '*(len('[PASSED]') + 2)
    if pixel:
        dim_type = 'Pixel'
    else:
        dim_type = 'Actual'
    suggested = False
    if len(dims_count) > 1:
        ret += get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' Intra Subject Image ('
                             + dim_type + ') Dimensions Check\n')
        suggested = True
        if cols is not None:
            suggested = cols[0]
        if verbose:
            ret += prefix + 'Image dimensions do not match in between images of the same subject\n'
            ret += prefix + 'Deepmedic requires all images of each subject to have the same dimensions.\n'
            ret += prefix + 'Subjects with different sized images:\n'
            ret += print_list(dims_count, prefix)

    else:
        ret += get_bold_text(get_html_colour(' [PASSED]', 'green') + ' Intra Subject Image ('
                             + dim_type + ') Dimensions Check\n')
        if verbose:
            ret += prefix + 'All images of the same subject have the same dimensions'

    if html:
        ret = text_to_html(ret)

    return ret, suggested


def dtype_check(dtype_count, dtype=sitk.sitkFloat32, verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 2)
    ret = ''
    suggested = None
    if len(dtype_count) > 1:
        ret += get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' Data Type Check\n')
        suggested = inv_pixel_type_dict[dtype]
        if verbose:
            ret += prefix + 'More than one data type\n'
            ret += prefix + 'We recommend resampling every image to ' + sitk.GetPixelIDValueAsString(dtype) + '\n'
            ret += prefix + 'Data Types Count:\n'
            ret += print_dict(dtype_count, prefix)
    else:
        if list(dtype_count.keys())[0] == sitk.GetPixelIDValueAsString(dtype):
            ret += get_bold_text(get_html_colour(' [PASSED]', 'green') + ' Data Type Check\n')
            if verbose:
                ret += prefix + 'Data Type: ' + str(list(dtype_count.keys())[0])
        else:
            ret += get_bold_text(get_html_colour('[WARNING]', 'orange', html) + ' Data Type Check</b>\n')
            ret += prefix + 'You might be using more memory than required storing your data\n'
            ret += prefix + 'This can increase the loading time.\n'
            ret += prefix + 'Consider resampling every image to ' + sitk.GetPixelIDValueAsString(dtype) + ' or smaller\n'
            suggested = inv_pixel_type_dict[dtype]
            if verbose:
                ret += prefix + 'Data Type: ' + str(list(dtype_count.keys())[0])

    if html:
        ret = text_to_html(ret)

    return ret, suggested


def dir_check(dir_count, verbose=True, html=False):
    prefix = ' '*(len('[PASSED]') + 2)
    ret = ''
    suggested = None
    if dir_count:
        ret += get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' Orientation Check\n')
        suggested = True
        if verbose:
            ret += prefix + str(dir_count) + ' images are not in Standard Radiology View\n'
            ret += prefix + 'We recommend reorienting the images'
    else:
        ret += get_bold_text(get_html_colour(' [PASSED]', 'green') + ' Orientation Check\n')
        ret += prefix + 'All images are in standard radiology view'

    if html:
        ret = text_to_html(ret)

    return ret, suggested


def run_checks(filelist, csv=False, pixs=False, dims=False, dtypes=False, dirs=False, sizes=False,
               intra_sizes=False, intra_dims=False,
               disable_tqdm=False, html=False, progress=None):

    suggested = {'intra_dimensions': None,
                 'intra_size': None,
                 'dimensions': None,
                 'size': None,
                 'spacing': None,
                 'dtype': None,
                 'direction': None,
                 'base_dir': None}

    image_cols = None

    if csv:
        filelist = pd.read_csv(filelist)
        valid_cols = ['Image', 'Mask', 'Target']
        image_cols = [col for col in filelist.columns if col.startswith('Channel_') or col in valid_cols]
        all_files = [item for sublist in filelist[image_cols].values.tolist() for item in sublist]
        suggested['base_dir'] = os.path.dirname(os.path.commonprefix(all_files))
        # check if all files exist
        missing_files = []
        for fn in all_files:
            if not os.path.isfile(fn):
                missing_files += [fn]
        if missing_files:
            prefix = ' ' * (len('[PASSED]') + 2)
            ret = get_bold_text(get_html_colour(' [FAILED]', 'red', html) + ' ' +
                                str(len(missing_files)) + ' FILES NOT FOUND. PLEASE FIX BEFORE CONTINUING.\n')
            for fn in missing_files:
                ret += prefix + get_bold_text('· ') + fn + '\n'

            ret = text_to_html(ret)

            if progress is not None:
                progress.bar.setMaximum(1)
                progress.bar.setValue(1)

            return ret, suggested

    if progress is not None:
        progress.bar.setMaximum(len(filelist))

    (intra_dims_count,
     intra_size_count,
     dims_count,
     spacing_count,
     dtype_count,
     direction_count,
     size_count) = get_image_dims_stats(filelist, do_dims=dims, do_pixs=pixs, do_dtypes=dtypes, do_size=sizes,
                                        do_intra_size=intra_sizes, do_intra_dims=intra_dims,
                                        disable_tqdm=disable_tqdm, tqdm_text='Running Image and Pixel Dimension Checks',
                                        progress=progress)
    ret = ''

    if intra_dims:
        aux_ret, suggested['intra_dimensions'] = intra_dims_check(intra_dims_count, pixel=True, html=html)
        ret += aux_ret
    if intra_sizes:
        aux_ret, suggested['intra_size'] = intra_dims_check(intra_dims_count, html=html, cols=image_cols)
        ret += aux_ret
    if pixs:
        aux_ret, suggested['spacing'] = pix_check(spacing_count, html=html)
        ret += aux_ret
    if dirs:
        aux_ret, suggested['direction'] = dir_check(direction_count, html=html)
        ret += aux_ret
    if dtypes:
        aux_ret, suggested['dtype'] = dtype_check(dtype_count, html=html)
        ret += aux_ret
    if dims:
        aux_ret, suggested['dimensions'] = dims_check(dims_count, pixel=True, html=html)
        ret += aux_ret
    if sizes:
        aux_ret, suggested['size'] = dims_check(size_count, html=html)
        ret += aux_ret

    return ret, suggested


def save_thumbnails(filelist, save_folder):
    for image_path in tqdm(filelist):
        image = NiftiImage(image_path)
        thumbnail_file = os.path.join(save_folder, image_path.split('/')[-1].split('.')[0])
        image.save_thumbnail(os.path.join(thumbnail_file + '.png'))


def resample_image_list(filelist, ref=None, origin=None, spacing=None, direction=None, size=None,
                        standard=False, suffix='', save_folder=None, thumbnails_folder=None, image_extension=None,
                        orientation=False, params=None, progress=None):
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
        os.makedirs(save_folder, exist_ok=True)

    if progress is not None:
        progress.bar.setMaximum(len(filelist))

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
            save_nifti(image.image, image_save_name + suffix + image_extension)
        if thumbnails_folder:
            thumbnail_file = os.path.join(thumbnails_folder, image_name.split('/')[-1])
            image.save_thumbnail(thumbnail_file + '.png')

        if progress is not None:
            progress.increase_value()


def resize_images(image_list, masks, new_size, save_path, tqdm_text='Resizing images', disable_tqdm=False):
    for image_path, mask_path in tqdm(zip(image_list, masks), desc=tqdm_text, disable=disable_tqdm,
                                      total=len(image_list)):
        image = NiftiImage(image_path, mask=mask_path)

        image.resize(new_size, centre_mass=True, use_mask=True)

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
