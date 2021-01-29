# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division


import collections
import numpy as np
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk


def apply_augmentations(augs, image, target=None, mask=None, wmaps=None):
    """
    Apply augmentations to input image
    Keyword arguments:
    augs -- list of augmentations (parent RandomAugmentation)
    image -- 4D array of image channels
    target -- 3D array of segmentation target
    mask -- 3D array of RoI mask
    wmaps -- 3D array of wmaps
    """
    if augs is not None:
        for aug in augs:
            image, target, mask, wmaps = aug(image, target, mask, wmaps)

    return image, target, mask, wmaps


# --------------------
# Augmentation Classes
# --------------------
class RandomAugmentation(object):
    """
    Abstract class for random patch augmentation, patch augmentation also works on full images
    __call__: When called a Augmentation should return an image and target and mask with the same shape
    as the input.
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target=None, mask=None, wmaps=None):
        if np.random.choice((True, False), p=(self.prob, 1. - self.prob)):
            return self.augment(image, target, mask, wmaps)
        else:
            return image, target, mask, wmaps

    def augment(self, image, target, mask, wmaps):
        raise NotImplementedError

    def get_attrs_str(self):
        attrs = vars(self)
        return '\t' + self.__class__.__name__ + '\n\t\t' + '\n\t\t'.join("%s: %s" % item for item in attrs.items())


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

    def augment(self, image, target, mask, wmaps):

        shape = image[0].shape[1:]
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
            wmaps.ravel()[indices].reshape(shape) if wmaps is not None else None
            for i in range(len(image)):
                image[i] = np.stack([channel.ravel()[indices].reshape(shape) for channel in image[i]])
        except IndexError:
            return image, target, mask, wmaps

        return image, target, mask, wmaps


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


class RandomAffineTransformation(RandomAugmentation):
    def __init__(self, prob, max_rot_xyz=(45., 45., 45.), max_scaling=.1, seed=None,
                 interp_order_imgs=1, interp_order_lbls=0, interp_order_roi=0,
                 interp_order_wmaps=1, boundary_mode='nearest', cval=0):
        super().__init__(prob)
        self.max_rot_xyz = max_rot_xyz
        self.max_scaling = max_scaling
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        # For calls.
        self.interp_order_imgs = interp_order_imgs
        self.interp_order_lbls = interp_order_lbls
        self.interp_order_roi = interp_order_roi
        self.interp_order_wmaps = interp_order_wmaps
        self.boundary_mode = boundary_mode
        self.cval = cval

    def _get_random_transformation(self):
        theta_x = self.rng.uniform(-self.max_rot_xyz[0], self.max_rot_xyz[0]) * np.pi / 180.
        rot_x = np.array([[np.cos(theta_x), -np.sin(theta_x), 0.],
                          [np.sin(theta_x), np.cos(theta_x), 0.],
                          [0., 0., 1.]])

        theta_y = self.rng.uniform(-self.max_rot_xyz[1], self.max_rot_xyz[1]) * np.pi / 180.
        rot_y = np.array([[np.cos(theta_y), 0., np.sin(theta_y)],
                          [0., 1., 0.],
                          [-np.sin(theta_y), 0., np.cos(theta_y)]])

        theta_z = self.rng.uniform(-self.max_rot_xyz[2], self.max_rot_xyz[2]) * np.pi / 180.
        rot_z = np.array([[1., 0., 0.],
                          [0., np.cos(theta_z), -np.sin(theta_z)],
                          [0., np.sin(theta_z), np.cos(theta_z)]])

        # Sample the scale (zoom in/out)
        # TODO: Non isotropic?
        scale = np.eye(3, 3) * self.rng.uniform(1 - self.max_scaling, 1 + self.max_scaling)

        # Affine transformation matrix.
        transformation_mtx = np.dot(scale, np.dot(rot_z, np.dot(rot_x, rot_y)))

        return transformation_mtx

    def _apply_transformation(self, image, transf_mtx, interp_order=2., boundary_mode='nearest', cval=0.):
        # image should be 3 dimensional (Height, Width, Depth). Not multi-channel.
        # interp_order: Integer. 1,2,3 for images, 0 for nearest neighbour on masks (GT & brainmasks)
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: float. value given to boundaries if mode is constant.
        assert interp_order in [0, 1, 2, 3]

        mode = boundary_mode
        if mode == 'min':
            cval = np.min(image)
            mode = 'constant'

        new_image = np.array(image, copy=True)

        # For recentering
        centre_coords = 0.5 * np.asarray(image.shape, dtype=np.int32)
        c_offset = centre_coords - centre_coords.dot(transf_mtx)

        new_image = scipy.ndimage.affine_transform(image,
                                                   transf_mtx.T,
                                                   c_offset,
                                                   order=interp_order,
                                                   mode=mode,
                                                   cval=cval)

        return new_image

    def augment(self, image, target, mask, wmaps):
        transf_mtx = self._get_random_transformation()

        for img_i in range(len(image)):
            image[img_i] = self._apply_transformation(image[img_i],
                                                      transf_mtx,
                                                      self.interp_order_imgs,
                                                      self.boundary_mode,
                                                      self.cval)

        if mask is not None:
            mask = self._apply_transformation(mask,
                                              transf_mtx,
                                              self.interp_order_roi,
                                              self.boundary_mode,
                                              self.cval)

        if target is not None:
            target = self._apply_transformation(target,
                                                transf_mtx,
                                                self.interp_order_lbls,
                                                self.boundary_mode,
                                                self.cval)

        if wmaps is not None:
            wmaps = self._apply_transformation(wmaps,
                                               transf_mtx,
                                               self.interp_order_wmaps,
                                               self.boundary_mode,
                                               self.cval)

        return image, target, mask, wmaps


class RandomHistogramDistortion(RandomAugmentation):
    def __init__(self, prob, shift=None, scale=None):
        super().__init__(prob)
        self.shift = shift
        self.scale = scale

    def augment(self, image, target, mask, wmaps):
        n_channs = image[0].shape[0]
        if self.shift is None:
            shift_per_chan = 0.
        elif self.shift['std'] != 0:  # np.random.normal does not work for an std==0.
            shift_per_chan = np.random.normal(self.shift['mu'], self.shift['std'], [n_channs, 1, 1, 1])
        else:
            shift_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * self.shift['mu']

        if self.scale is None:
            scale_per_chan = 1.
        elif self.scale['std'] != 0:
            scale_per_chan = np.random.normal(self.scale['mu'], self.scale['std'], [n_channs, 1, 1, 1])
        else:
            scale_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * self.scale['mu']

        # Intensity augmentation
        for path_idx in range(len(image)):
            image[path_idx] = (image[path_idx] + shift_per_chan) * scale_per_chan

        return image, target, mask, wmaps


class RandomFlip(RandomAugmentation):
    def __init__(self, prob=1., prob_flip_axes=None):
        super().__init__(prob)
        if prob_flip_axes is None:
            self.prob_flip_axes = tuple([1] * 3)
        else:
            self.prob_flip_axes = prob_flip_axes

    def augment(self, image, target, mask, wmaps):
        for axis_idx in range(len(image[0].shape) - 1):  # 3 dims ( -1 because dim [0] refers to the channels)
            flip = np.random.choice(a=(True, False), size=1,
                                    p=(self.prob_flip_axes[axis_idx], 1. - self.prob_flip_axes[axis_idx]))
            if flip:
                for path_idx in range(len(image)):
                    image[path_idx] = np.flip(image[path_idx], axis=axis_idx + 1)  # + 1 because dim [0] is channels.
                target = np.flip(target, axis=axis_idx) if target is not None else None
                mask = np.flip(mask, axis=axis_idx) if mask is not None else None
                wmaps = np.flip(wmaps, axis=axis_idx) if wmaps is not None else None

        return image, target, mask, wmaps


class RandomRotation90(RandomAugmentation):
    def __init__(self, prob=1., prob_rot_90=None):
        super().__init__(prob)
        if prob_rot_90 is None:
            self.prob_rot_90 = {
                'xy': {'0': 1, '90': 1, '180': 1, '270': 1},
                'yz': {'0': 1, '90': 1, '180': 1, '270': 1},
                'xz': {'0': 1, '90': 1, '180': 1, '270': 1}
            }
        else:
            self.prob_rot_90 = prob_rot_90

    def augment(self, image, target, mask, wmaps):
        for key, plane_axes in zip(['xy', 'yz', 'xz'], [(0, 1), (1, 2), (0, 2)]):
            probs_plane = self.prob_rot_90[key]

            if probs_plane is None:
                continue

            assert len(probs_plane) == 4  # rotation 0, rotation 90 degrees, 180, 270.
            # +1 cause [0] is channel. Image/patch must be isotropic.
            # assert image.shape[1 + plane_axes[0]] == image.shape[1 + plane_axes[1]]

            # Normalize probs
            sum_p = probs_plane['0'] + probs_plane['90'] + probs_plane['180'] + probs_plane['270']
            if sum_p == 0:
                continue
            for rot_k in probs_plane:
                probs_plane[rot_k] /= sum_p  # normalize p to 1.

            p_rot_90_x0123 = (probs_plane['0'], probs_plane['90'], probs_plane['180'], probs_plane['270'])
            rot_90_xtimes = np.random.choice(a=(0, 1, 2, 3), size=1, p=p_rot_90_x0123)
            for path_idx in range(len(image)):
                image[path_idx] = np.rot90(image[path_idx], k=rot_90_xtimes,
                                           axes=[axis + 1 for axis in plane_axes])  # + 1 cause [0] is channels.
            target = np.rot90(target, k=rot_90_xtimes, axes=plane_axes) if target is not None else None
            mask = np.rot90(mask, k=rot_90_xtimes, axes=plane_axes) if mask is not None else None
            wmaps = np.rot90(wmaps, k=rot_90_xtimes, axes=plane_axes) if wmaps is not None else None

        return image, target, mask, wmaps
