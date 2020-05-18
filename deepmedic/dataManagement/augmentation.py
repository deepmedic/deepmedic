import numpy as np
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


class RandomAugmentation(object):
    """
    Abstract class for random patch augmentation, patch augmentation also works on full images
    __call__: When called a Augmentation should return an image and target and mask with the same shape
    as the input.
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target=None, mask=None):
        if np.random.choice((True, False), p=(self.prob, 1. - self.prob)):
            return self.augment(image, target, mask)
        else:
            return image, target, mask

    def augment(self, image, target, mask):
        raise NotImplementedError


class RandomPatchRotation(RandomAugmentation):
    def __init__(self, prob, allowed_planes, rotations=(1, 2, 3)):
        super().__init__(prob)
        self.allowed_planes = allowed_planes
        self.rotations = rotations

    def augment(self, image, target, mask):
        k = np.random.choice(self.rotations, len(self.allowed_planes))
        for i, axes in enumerate(self.allowed_planes):
            axes = np.random.choice(axes, 2, replace=False)  # direction of rotation
            image = np.rot90(image, k=k[i], axes=tuple(a + 1 for a in axes))
            target = np.rot90(target, k=k[i], axes=axes) if target is not None else None
            mask = np.rot90(mask, k=k[i], axes=axes) if mask is not None else None

        return image, target, mask


class RandomPatchFlip(RandomAugmentation):
    def __init__(self, prob, allowed_axis):
        super().__init__(prob)
        self.allowed_axes = allowed_axis

    def augment(self, image, target, mask):
        for axis in self.allowed_axes:
            image = np.flip(image, axis=axis + 1)
            target = np.flip(target, axis=axis) if target is not None else None
            mask = np.flip(mask, axis=axis) if mask is not None else None

        return image, target, mask


class RandomHistogramDeformation(RandomAugmentation):
    def __init__(self, prob, shift_std=0.05, scale_std=0.01, allow_mirror=False):
        super().__init__(prob)
        self.shift_std = shift_std
        self.scale_std = scale_std
        self.allow_mirror = allow_mirror

    def augment(self, image, target, mask):
        num_channels = image.shape[0]
        shift = np.random.uniform(0, self.shift_std, num_channels)
        scale = np.random.normal(1, self.scale_std, num_channels)
        if self.allow_mirror:
            scale *= np.random.choice((-1, 1))
        image = (image.T * scale).T
        image = (image.T + shift).T
        return image, target, mask


class RandomGammaCorrection(RandomAugmentation):
    def __init__(self, prob, range_min=-1., range_max=1., gamma_std=.1):
        super().__init__(prob)
        self.range_min = range_min
        self.range_max = range_max
        self.gamma_std = gamma_std

    def augment(self, image, target, mask):
        num_channels = image.shape[0]
        # gamma correction must be performed in the range of 0 to 1
        image = (image - self.range_min) / (self.range_max - self.range_min)
        gamma = np.random.normal(1, self.gamma_std, num_channels)
        image = np.power(image.T, gamma).T
        image = image * (self.range_max - self.range_min) + self.range_min
        return image, target, mask


class NonLinearRandomHistogramDeformation(RandomHistogramDeformation):
    def __init__(self, prob, range_min=-1., range_max=1., num_segments=5, shift_std=0.25, scale_std=0.25,
                 allow_mirror=True):
        super().__init__(prob, shift_std, scale_std, allow_mirror)
        segments = np.linspace(range_min, range_max, num_segments + 1)
        segments[-1] += 1  # ensure whole array gets distorted
        self.segments = segments

    def augment(self, image, target, mask):
        new_image = np.zeros_like(image)
        for i, _ in enumerate(self.segments[:-1]):
            indices = np.logical_and(image >= self.segments[i], image < self.segments[i + 1])
            image_segment, _, _ = super().augment(image, target, mask)
            new_image[indices] = image_segment[indices]

        return new_image, target, mask


class RandomElasticDeformation(RandomAugmentation):
    """
    alpha: The amplitude of the noise;
    prob: Probability of deformation occurring
    noise_shape: Shape of the deformation field from which to sample patches from (must be larger than input_shape)
    num_maps Number of different noise maps to generate
    """

    def __init__(self, prob, alpha, noise_shape, num_maps=3):
        super().__init__(prob)
        self.alpha = alpha
        self.num_maps = num_maps
        self.noise_shape = noise_shape
        self.deformation_fields = [np.round(self.get_1d_displacement_field(self.noise_shape)).astype(np.int32)
                                   for _ in range(self.num_maps)]
        self.patch_shape = None
        self.grid = None

    def get_1d_displacement_field(self, shape):
        raise NotImplementedError

    def get_displacement_field(self, patch_shape):
        dx = [self.deformation_fields[i] for i in np.random.choice(len(self.deformation_fields), 3)]
        starts = [[np.random.randint(s - ps + 1) for s, ps in zip(self.noise_shape, patch_shape)] for _ in range(3)]
        slices = [tuple(slice(s, s + ps, 1) for s, ps in zip(start, patch_shape)) for start in starts]
        return [dx[i][slices[i]] for i in range(3)]

    def augment(self, image, target, mask):

        shape = image.shape[1:]
        if shape != self.patch_shape:
            self.grid = [g.astype(np.int32) for g in np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')]
            self.patch_shape = shape

        dx = self.get_displacement_field(shape)

        indices = sum(np.clip((x_i + dx_i), a_min=0, a_max=s_i - 1).reshape(-1, 1) *
                      np.prod(shape[(i + 1):]).astype(np.int32)
                      for i, (x_i, dx_i, s_i) in enumerate(zip(self.grid, dx, shape)))
        try:
            target = target.ravel()[indices].reshape(shape) if target is not None else None
            mask = mask.ravel()[indices].reshape(shape) if mask is not None else None
            image = np.stack([channel.ravel()[indices].reshape(shape) for channel in image])
        except IndexError:
            return image, target, mask

        return image, target, mask


class RandomElasticDeformationSimard2003(RandomElasticDeformation):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    def __init__(self, prob, sigma, alpha, noise_shape, num_maps=3):
        self.sigma = sigma
        super().__init__(prob, alpha, noise_shape, num_maps)

    def get_1d_displacement_field(self, shape):
        dx = np.random.rand(*shape) * 2 - 1
        dx = gaussian_filter(dx, self.sigma, mode="nearest") * self.alpha
        return dx


class RandomElasticDeformationCoarse(RandomElasticDeformationSimard2003):
    def __init__(self, prob, sigma, coarseness, alpha, noise_shape, num_maps=3):
        self.coarseness = coarseness
        super().__init__(prob, sigma, alpha, noise_shape, num_maps)

    def get_1d_displacement_field(self, shape):
        coarse_shape = tuple(s // c + bool(s % c) for s, c in zip(shape, self.coarseness))
        dx = np.random.rand(*coarse_shape) * 2 - 1
        dx = np.kron(dx, np.ones(shape=self.coarseness))
        dx = gaussian_filter(dx, self.sigma, mode="nearest") * self.alpha
        dx = dx[tuple(slice(0, s, 1) for s in shape)]
        return dx


class RandomElasticDeformationCoarsePerlinNoise(RandomElasticDeformation):

    def __init__(self, prob, period, alpha, noise_shape, num_maps=3):
        self.period = period
        super().__init__(prob, alpha, noise_shape, num_maps)

    def get_1d_displacement_field(self, shape):
        new_shape = tuple((s // c + bool(s % c) + bool(c == 1)) * c for s, c in zip(shape, self.period))
        dx = self.generate_fractal_noise_3d(new_shape, self.period)
        dx = dx[tuple(slice(0, s, 1) for s in shape)] * self.alpha
        return dx

    # https://github.com/pvigier/perlin-numpy
    def generate_fractal_noise_3d(self, shape, res, octaves=1, persistence=0.5):
        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.generate_perlin_noise_3d(shape, (
                frequency * res[0], frequency * res[1], frequency * res[2]))
            frequency *= 2
            amplitude *= persistence
        return noise

    @staticmethod
    def generate_perlin_noise_3d(shape, res):
        def f(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
        d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
        grid = grid.transpose(1, 2, 3, 0) % 1
        # Gradients
        random_state = np.random.RandomState(843)
        theta = 2 * np.pi * random_state.rand(res[0] + 1, res[1] + 1, res[2] + 1)
        phi = 2 * np.pi * random_state.rand(res[0] + 1, res[1] + 1, res[2] + 1)
        gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)

        g000 = gradients[0:-1, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g100 = gradients[1:, 0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g010 = gradients[0:-1, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g110 = gradients[1:, 1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g001 = gradients[0:-1, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g101 = gradients[1:, 0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g011 = gradients[0:-1, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g111 = gradients[1:, 1:, 1:].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)

        # Ramps
        n000 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
        n100 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
        n010 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
        n110 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
        n001 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
        n101 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
        n011 = np.sum(np.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
        n111 = np.sum(np.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
        # Interpolation
        t = f(grid)
        n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
        n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
        n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
        n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
        n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
        n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11

        return (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1

# class RandomAffineTransformation(Augmentation):
#     def __init__(self, prob, max_xy_rot=np.pi / 4, max_xz_rot=np.pi / 6, max_yz_rot=np.pi / 6, max_scaling=.1,
#                  default_channel_value=-1., default_target_value=0., is_target_discrete=True):
#         self.prob =  prob
#         self.max_xy_rot = max_xy_rot
#         self.max_xz_rot = max_xz_rot
#         self.max_yz_rot = max_yz_rot
#         self.max_scaling = max_scaling
#         self.default_channel_value = default_channel_value
#         self.default_target_value = default_target_value
#         self.is_target_discrete = is_target_discrete
#
#     @staticmethod
#     def resample(channel, transform, interpolation, default_value):
#         im = sitk.GetImageFromArray(channel)
#         return sitk.GetArrayFromImage(sitk.Resample(im, im, transform, interpolation, default_value))
#
#     def __call__(self, image, target=None, mask=None):
#         if np.random.random_sample() > self.prob:
#             return image, target, sampling_mask
#
#         transform = sitk.AffineTransform(3)
#
#         theta_xy = np.random.uniform(-self.max_xy_rot, self.max_xy_rot)
#         rot_xy = np.array(
#             [[np.cos(theta_xy), np.sin(theta_xy), 0], [-np.sin(theta_xy), np.cos(theta_xy), 0], [0, 0, 1]])
#
#         theta_xz = np.random.uniform(-self.max_xz_rot, self.max_xz_rot)
#         rot_xz = np.array(
#             [[np.cos(theta_xz), 0, np.sin(theta_xz)], [0, 1, 0], [-np.sin(theta_xy), 0, np.cos(theta_xy)]])
#
#         theta_yz = np.random.uniform(-self.max_yz_rot, self.max_yz_rot)
#         rot_yz = np.array(
#             [[1, 0, 0], [0, np.cos(theta_yz), np.sin(theta_yz)], [0, -np.sin(theta_yz), np.cos(theta_yz)]])
#
#         scale = np.eye(3, 3) * np.random.uniform(1 - self.max_scaling, 1 + self.max_scaling)
#
#         matrix = np.dot(scale, np.dot(np.dot(rot_xy, rot_xz), rot_yz))
#         transform.SetMatrix(matrix.ravel())
#
#         image = np.array(
#             [self.resample(channel, transform, sitk.sitkLinear, self.default_channel_value) for channel in image])
#         target = np.array(self.resample(target, transform, sitk.sitkNearestNeighbor, self.default_target_value))
#         sampling_mask = np.array(self.resample(sampling_mask.astype(np.uint8), transform, sitk.sitkNearestNeighbor, 0))
#
#         return image, target, sampling_mask
# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import collections
import numpy as np
import scipy.ndimage


# Main function to call:
def augment_imgs_of_case(channels, gt_lbls, roi_mask, wmaps_per_cat, prms):
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]. Can be None.
    # roi_mask: np array of shape [x,y,z]. Can be None.
    # wmaps_per_cat: List of np.arrays (floats or ints), weightmaps for sampling. Can be None.
    # prms: None (for no augmentation) or Dictionary with parameters of each augmentation type. }
    if prms is not None:
        (channels,
        gt_lbls,
        roi_mask,
        wmaps_per_cat) = random_affine_deformation( channels,
                                                    gt_lbls,
                                                    roi_mask,
                                                    wmaps_per_cat,
                                                    prms['affine'] )
    return channels, gt_lbls, roi_mask, wmaps_per_cat


def random_affine_deformation(channels, gt_lbls, roi_mask, wmaps_l, prms):
    if prms is None:
        return channels, gt_lbls, roi_mask, wmaps_l
    
    augm = AugmenterAffine(prob = prms['prob'],
                           max_rot_xyz = prms['max_rot_xyz'],
                           max_scaling = prms['max_scaling'],
                           seed = prms['seed'])
    transf_mtx = augm.roll_dice_and_get_random_transformation()
    assert transf_mtx is not None
    
    channels  = augm(images_l = channels,
                     transf_mtx = transf_mtx,
                     interp_orders = prms['interp_order_imgs'],
                     boundary_modes = prms['boundary_mode'])
    (gt_lbls,
    roi_mask) = augm(images_l = [gt_lbls, roi_mask],
                     transf_mtx = transf_mtx,
                     interp_orders = [prms['interp_order_lbls'], prms['interp_order_roi']],
                     boundary_modes = prms['boundary_mode'])
    wmaps_l   = augm(images_l = wmaps_l,
                     transf_mtx = transf_mtx,
                     interp_orders = prms['interp_order_wmaps'],
                     boundary_modes = prms['boundary_mode'])

    return channels, gt_lbls, roi_mask, wmaps_l



class AugmenterParams(object):
    # Parent class, for parameters of augmenters.
    def __init__(self, prms):
        # prms: dictionary
        self._prms = collections.OrderedDict()
        self._set_from_dict(prms)
    
    def __str__(self):
        return str(self._prms)
    
    def __getitem__(self, key): # overriding the [] operator.
        # key: string.
        return self._prms[key] if key in self._prms else None
    
    def __setitem__(self, key, item): # For instance[key] = item assignment
        self._prms[key] = item

    def _set_from_dict(self, prms):
        if prms is not None:
            for key in prms.keys():
                self._prms[key] = prms[key]
                
                
class AugmenterAffineParams(AugmenterParams):
    def __init__(self, prms):
        # Default values.
        self._prms = collections.OrderedDict([ ('prob', 0.0),
                                               ('max_rot_xyz', (45., 45., 45.)),
                                               ('max_scaling', .1),
                                               ('seed', None),
                                               # For calls.
                                               ('interp_order_imgs', 1),
                                               ('interp_order_lbls', 0),
                                               ('interp_order_roi', 0),
                                               ('interp_order_wmaps', 1),
                                               ('boundary_mode', 'nearest'),
                                               ('cval', 0.) ])
        # Overwrite defaults with given.
        self._set_from_dict(prms)
    
    def __str__(self):
        return str(self._prms)


class AugmenterAffine(object):
    def __init__(self, prob, max_rot_xyz, max_scaling, seed=None):
        self.prob = prob # Probability of applying the transformation.
        self.max_rot_xyz = max_rot_xyz
        self.max_scaling = max_scaling
        self.rng = np.random.RandomState(seed)

    def roll_dice_and_get_random_transformation(self):
        if self.rng.random_sample() > self.prob:
            return -1 # No augmentation
        else:
            return self._get_random_transformation() #transformation for augmentation
        
    def _get_random_transformation(self):
        theta_x = self.rng.uniform(-self.max_rot_xyz[0], self.max_rot_xyz[0]) * np.pi / 180.
        rot_x = np.array([ [np.cos(theta_x), -np.sin(theta_x), 0.],
                           [np.sin(theta_x), np.cos(theta_x), 0.],
                           [0., 0., 1.]])
        
        theta_y = self.rng.uniform(-self.max_rot_xyz[1], self.max_rot_xyz[1]) * np.pi / 180.
        rot_y = np.array([ [np.cos(theta_y), 0., np.sin(theta_y)],
                           [0., 1., 0.],
                           [-np.sin(theta_y), 0., np.cos(theta_y)]])
        
        theta_z = self.rng.uniform(-self.max_rot_xyz[2], self.max_rot_xyz[2]) * np.pi / 180.
        rot_z = np.array([ [1., 0., 0.],
                           [0., np.cos(theta_z), -np.sin(theta_z)],
                           [0., np.sin(theta_z), np.cos(theta_z)]])
        
        # Sample the scale (zoom in/out)
        # TODO: Non isotropic?
        scale = np.eye(3, 3) * self.rng.uniform(1 - self.max_scaling, 1 + self.max_scaling)
        
        # Affine transformation matrix.
        transformation_mtx = np.dot( scale, np.dot(rot_z, np.dot(rot_x, rot_y)) )
        
        return transformation_mtx

    def _apply_transformation(self, image, transf_mtx, interp_order=2., boundary_mode='nearest', cval=0.):
        # image should be 3 dimensional (Height, Width, Depth). Not multi-channel.
        # interp_order: Integer. 1,2,3 for images, 0 for nearest neighbour on masks (GT & brainmasks)
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: float. value given to boundaries if mode is constant.
        assert interp_order in [0,1,2,3]
        
        mode = boundary_mode
        if mode == 'min':
            cval = np.min(image)
            mode = 'constant'
        
        # For recentering
        centre_coords = 0.5 * np.asarray(image.shape, dtype=np.int32)
        c_offset = centre_coords - centre_coords.dot( transf_mtx )
        
        new_image = scipy.ndimage.affine_transform( image,
                                                    transf_mtx.T,
                                                    c_offset,
                                                    order=interp_order,
                                                    mode=mode,
                                                    cval=cval )
        return new_image
    
    def __call__(self, images_l, transf_mtx, interp_orders, boundary_modes, cval=0.):
        # images_l : List of images, or an array where first dimension is over images (eg channels).
        #            An image (element of the var) can be None, and it will be returned unchanged.
        #            If images_l is None, then returns None.
        # transf_mtx: Given (from get_random_transformation), -1, or None.
        #             If -1, no augmentation/transformation will be done.
        #             If None, new random will be made.
        # intrp_orders : Int or List of integers. Orders of bsplines for interpolation, one per image in images_l.
        #                Suggested: 3 for images. 1 is like linear. 0 for masks/labels, like NN.
        # boundary_mode = String or list of strings. 'constant', 'min', 'nearest', 'mirror...
        # cval: single float value. Value given to boundaries if mode is 'constant'.
        if images_l is None:
            return None
        if transf_mtx is None: # Get random transformation.
            transf_mtx = self.roll_dice_and_get_random_transformation()
        if not isinstance(transf_mtx, np.ndarray) and transf_mtx == -1: # Do not augment
            return images_l
        # If scalars/string was given, change it to list of scalars/strings, per image.
        if isinstance(interp_orders, int):
            interp_orders = [interp_orders] * len(images_l)
        if isinstance(boundary_modes, str):
            boundary_modes = [boundary_modes] * len(images_l)
        
        # Deform images.
        for img_i, int_order, b_mode in zip(range(len(images_l)), interp_orders, boundary_modes):
            if images_l[img_i] is None:
                pass # Dont do anything. Let it be None.
            else:
                images_l[img_i] = self._apply_transformation(images_l[img_i],
                                                             transf_mtx,
                                                             int_order,
                                                             b_mode,
                                                             cval)
        return images_l



############# Currently not used ####################

# DON'T use on patches. Only on images. Cause I ll need to find min and max intensities, to move to range [0,1]
def random_gamma_correction(channels, gamma_std=0.05):
    # Gamma correction: I' = I^gamma
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # IMPORTANT: Does not work if intensities go to negatives.
    if gamma_std is None or gamma_std == 0.:
        return channels
    
    n_channs = channels[0].shape[0]
    gamma = np.random.normal(1, gamma_std, [n_channs,1,1,1])
    for path_idx in range(len(channels)):
        assert np.min(channels[path_idx]) >= 0.
        channels[path_idx] = np.power(channels[path_idx], 1.5, dtype='float32')
        
    return channels



            
