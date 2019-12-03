from __future__ import print_function

import numpy as np
from PIL import Image
import SimpleITK as sitk
import math
from scipy import ndimage


pixel_type_dict = {"uint8": sitk.sitkUInt8,
                   "int8": sitk.sitkInt8,
                   "uint16": sitk.sitkUInt16,
                   "int16": sitk.sitkInt16,
                   "uint32": sitk.sitkUInt32,
                   "int32": sitk.sitkInt32,
                   "uint64": sitk.sitkUInt64,
                   "int64": sitk.sitkInt64,
                   "float32": sitk.sitkFloat32,
                   "float64": sitk.sitkFloat64}


def pixel_type_to_sitk(pixel_type):
    try:
        return pixel_type_dict[pixel_type]
    except KeyError:
        pass
    return None


def get_nifti_reader(filename):
    reader = sitk.ImageFileReader()

    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()

    reader.ReadImageInformation()

    return reader


def save_nifti(image, filename):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(image)


def get_new_size(size, new_pix_dims, old_pix_dims):
    return tuple([int(x) for x in np.array(size) * (np.array(old_pix_dims) / np.array(new_pix_dims))])


def get_new_origin(origin, new_pix_dims, old_pix_dims):
    return origin  # tuple([x for x in np.array(origin) * (np.array(old_pix_dims) / np.array(new_pix_dims))])


def reorient_params(direction, size, spacing, origin):
    """Reorients an image to standard radiology view."""

    dir = np.array(direction).reshape(len(size), -1)
    ind = np.argmax(np.abs(dir), axis=0)
    new_size = np.array(size)[ind]
    new_spacing = np.array(spacing)[ind]
    new_extent = new_size * new_spacing
    new_dir = dir[:, ind]

    flip = np.diag(new_dir) < 0
    flip_diag = flip * -1
    flip_diag[flip_diag == 0] = 1
    flip_mat = np.diag(flip_diag)

    new_origin = np.array(origin) + np.matmul(new_dir, (new_extent * flip))
    new_dir = np.matmul(new_dir, flip_mat)

    needs_reorient = sum(flip) > 0

    return needs_reorient, tuple(new_dir.flatten()), tuple(new_size.astype(int).tolist()), tuple(new_spacing), tuple(new_origin)


def get_min_size_idxs(image, dims, size):
    min_i_list = np.zeros(dims)
    max_i_list = np.zeros(dims)
    for dim in range(dims):
        if not size[-(dim + 1)] == 1:
            _sum = np.sum(sitk.GetArrayFromImage(image.open()), axis=tuple([a for a in range(dims) if not a == dim]))

            # print(self.get_size()[-(dim + 1)])
            if not len(_sum) == 1:
                for i in range(1, size[-(dim + 1)]):
                    if _sum[i] > 1:
                        min_i_list[-(dim + 1)] = i
                        break
                for i in range(1, size[-(dim + 1)]):
                    if _sum[-i] > 1:
                        max_i_list[[-(dim + 1)]] = size[-(dim + 1)] - i
                        break
    return min_i_list, max_i_list


def greater_than(size1, size2):
    compare = []
    for i in range(len(size1)):
        compare.append(size1[i] > size2[i])

    return compare


class NiftiImage(object):

    def __init__(self, filename):
        self.image_reader = get_nifti_reader(filename)
        self.image = None
        self.reader = self.image_reader

    def update_reader(self, filename):
        self.reader = get_nifti_reader(filename)

    def open(self):
        if self.image is None:
            self.image = self.reader.Execute()
            self.reader = self.image
        return self.image

    def get_metadata_dict(self):
        metadata = {}
        for k in self.reader.GetMetaDataKeys():
            metadata[k] = self.reader.GetMetaData(k)
        return metadata

    def get_num_dims(self):
        return self.reader.GetDimension()

    def get_size(self):
        return self.reader.GetSize()

    def get_spacing(self):
        return self.reader.GetSpacing()

    def get_origin(self):
        return self.reader.GetOrigin()

    def get_direction(self):
        return self.reader.GetDirection()

    def get_pixel_type(self):
        return self.reader.GetPixelIDValue()

    def get_pixel_type_string(self):
        dtype = self.get_pixel_type()
        return sitk.GetPixelIDValueAsString(dtype)

    def get_resample_parameters(self):
        return self.get_size(), self.get_spacing(), self.get_direction(), self.get_origin()

    def get_header_keys(self):
        return self.reader.GetMetaDataKeys()

    def is_in_std_radiology_view(self):
        dir = np.array(self.get_direction()).reshape(len(self.get_size()), -1)
        ind = np.argmax(np.abs(dir), axis=0)
        new_dir = dir[:, ind]

        flip = np.diag(new_dir) < 0

        return sum(flip) <= 0

    def apply_resample(self, origin, spacing, direction, size, interpolator=sitk.sitkLinear):

        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(interpolator)
        resample.SetOutputDirection(direction)
        resample.SetOutputOrigin(origin)
        resample.SetOutputSpacing(spacing)
        resample.SetSize((size[0], size[1], max(size[2], 1)))

        return resample.Execute(self.open())

    def reorient(self):
        needs_reorient, direction, size, spacing, origin = reorient_params(self.get_direction(), self.get_size(),
                                                                           self.get_spacing(), self.get_origin())
        if needs_reorient:
            self.image = self.apply_resample(origin, spacing, direction, size)
            self.reader = self.image

    def get_mask(self, min_intensity, max_intensity, filename=None):
        self.open()
        image = sitk.GetArrayFromImage(self.open())
        mask = np.zeros(image.shape)
        if min_intensity is None:
            min_intensity = min(image.flatten())

        if max_intensity is None:
            max_intensity = max(image.flatten())

        mask[(min_intensity < image) & (image < max_intensity)] = 1

        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_image.SetOrigin(self.get_origin())
        mask_image.SetSpacing(self.get_spacing())
        mask_image.SetDirection(self.get_direction())

        if filename:
            save_nifti(mask_image, filename)

        return mask_image

    def resample(self, origin=None, spacing=None, direction=None, size=None, standard=False,
                 save=False, filename=None, copy=False, ref_image=None):
        # get transformation parameters
        if ref_image:
            size_rsp, spacing_rsp, direction_rsp, origin_rsp = ref_image.get_resample_parameters()
            if not size:
                size = size_rsp
            if not spacing:
                spacing = spacing_rsp
            if not direction:
                direction = direction_rsp
            if not origin:
                origin = origin_rsp  # get_new_origin(self.get_origin(), spacing, self.get_spacing())
                # origin = origin_rsp
        else:
            if spacing is None:
                if standard:
                    spacing = (1., 1., 1.)
                else:
                    spacing = self.get_spacing()
            if size is None:
                size = get_new_size(self.get_size(), spacing, self.get_spacing())
            num_dims = len(size)
            if origin is None:
                if standard:
                    origin = np.zeros(num_dims)
                else:
                    origin = self.get_origin()  # get_new_origin(self.get_origin(), spacing, self.get_spacing())
            if direction is None:
                if standard:
                    direction = np.identity(num_dims).flatten()
                else:
                    direction = self.get_direction()

        # apply transformation
        resampled = self.apply_resample(origin, spacing, direction, size)

        # save/update
        if save:
            save_nifti(resampled, filename)
        if not copy:
            self.image = resampled
            self.reader = self.image

        return resampled

    def save_thumbnail(self, filename, thumbnail_size=(128, 128), max_slice=False,
                       min_intensity=None, max_intensity=None):
        size = self.get_size()
        image = sitk.GetArrayFromImage(self.open())
        if max_slice:
            image_bool = image > 0
            max_val = 0
            max_i = None
            for i in range(size[2]):
                sum_ = np.sum(image_bool[i, :, :])
                if sum_ > max_val:
                    max_val = sum_
                    max_i = i
        else:
            max_i = int(size[2]/2)
        img_slice = image[max_i, :, :]
        if min_intensity or max_intensity:
            np.clip(image, min_intensity, max_intensity, out=image)
        # img_slice = np.flip(image[max_i, :, :], (0, 1))
        img_slice = (img_slice - np.min(img_slice)) * 255.0 / (np.max(img_slice) - np.min(img_slice))
        im = Image.fromarray(img_slice)
        im = im.convert('L')
        im.thumbnail(thumbnail_size)
        im.save(filename)

    def save(self, filename):
        save_nifti(self.open(), filename)

    def get_min_size(self):
        min_i_list, max_i_list = get_min_size_idxs(self, self.get_num_dims(), self.get_size())
        min_size = max_i_list - min_i_list + 1
        return min_size

    def crop(self, min_crop, max_crop):
        self.open()
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetLowerBoundaryCropSize(min_crop)
        crop_filter.SetUpperBoundaryCropSize(max_crop)
        self.image = crop_filter.Execute(self.image)

    def pad_constant(self, constant, min_pad, max_pad):
        self.open()
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetConstant(constant)
        pad_filter.SetPadLowerBound(min_pad)
        pad_filter.SetPadUpperBound(max_pad)
        self.image = pad_filter.Execute(self.image)

    def get_centre_mass(self, mask):
        if mask:
            image_nii = mask.open()
        else:
            image_nii = self.open()
        image = sitk.GetArrayFromImage(image_nii)
        centre_mass_rev = ndimage.measurements.center_of_mass(image)
        centre_mass_rev = [centre_mass_rev[i] if not np.isnan(centre_mass_rev[i]) else image.shape[i]/2
                           for i in range(len(centre_mass_rev))]
        return list(map(math.floor, reversed(centre_mass_rev)))

    def resize(self, size, mask=None, centre_mass=False, crop_mask=False):
        self.open()
        if centre_mass:
            centre_mass_idxs = self.get_centre_mass(mask)
            min_i_list = centre_mass_idxs
            max_i_list = centre_mass_idxs
            mask_size = np.ones(len(max_i_list))
        else:
            if mask:
                mask.open()
                min_i_list, max_i_list = get_min_size_idxs(mask, mask.get_num_dims(), mask.get_size())
                mask_size = max_i_list - min_i_list
            else:
                mask_size = np.array(self.get_size())
                min_i_list = np.zeros(3)
                max_i_list = mask_size - 1

        total_margin = np.array(list(size)) - mask_size
        min_i_resized = min_i_list - np.array([math.floor(margin / 2) for margin in total_margin])
        max_i_resized = max_i_list + np.array([math.ceil(margin / 2) for margin in total_margin])

        needs_cropping = bool(sum(min_i_resized > 0) + sum(max_i_resized + 1 < np.array(list(self.get_size()))))
        needs_padding = bool(sum(min_i_resized < 0) + sum(max_i_resized + 1 > np.array(list(self.get_size()))))

        if needs_cropping:  # crop
            min_crop = [max(int(a), 0) for a in min_i_resized]
            max_crop = [max(int(size_a - a - 1), 0) for a, size_a in zip(max_i_resized, list(self.get_size()))]

            if crop_mask:
                mask.crop(min_crop, max_crop)
            self.crop(min_crop, max_crop)

        if needs_padding:  # pad
            min_pad = [max(int(-a), 0) for a in min_i_resized]
            max_pad = [max(int(a - size_a + 1), 0) for a, size_a in zip(max_i_resized, list(self.get_size()))]
            if crop_mask:
                mask.pad_constant(0, min_pad, max_pad)
            self.pad_constant(-1000, min_pad, max_pad)

        self.reader = self.image

        if crop_mask:
            return mask
        else:
            return None

    def change_pixel_type(self, pixel_type):
        pixel_type_sitk = pixel_type_to_sitk(pixel_type)
        if pixel_type_sitk is None:
            self.open()
            return None
        filter = sitk.CastImageFilter()
        filter.SetOutputPixelType(pixel_type_sitk)
        self.image = filter.Execute(self.open())
