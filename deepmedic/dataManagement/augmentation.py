import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom, rotate
import warnings
import SimpleITK as sitk
import scipy.ndimage


def crop_rescale(image, slc, scale, order):
    image = image[tuple(slc)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = zoom(image, scale, order=order)
    return image


def to_array(a, ndim):
    if not isinstance(a, list):
        return a * np.ones(ndim)
    else:
        return a


def apply_augmentations(augs, image, target=None, mask=None, wmaps=None):
    if augs is not None:
        # if image.__class__ == list:
        #     print('list')
        #     for i in range(len(image)):
        #         for aug in augs:
        #             image_tmp, target_tmp, mask_tmp, wmaps_tmp = aug(image[i],
        #                                                              target[i] if target is not None else None,
        #                                                              mask[i] if mask is not None else None,
        #                                                              wmaps[i] if wmaps is not None else None)
        #             image[i] = image_tmp
        #             if target is not None:
        #                 target[i] = target_tmp
        #             if mask is not None:
        #                 mask[i] = mask_tmp
        #             if wmaps is not None:
        #                 wmaps[i] = wmaps_tmp
        # else:
        #     print('not list')
        for aug in augs:
            image, target, mask, wmaps = aug(image, target, mask, wmaps)

    return image, target, mask, wmaps


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


class RandomPatchRotation(RandomAugmentation):
    def __init__(self, prob, allowed_planes, rotations=(1, 2, 3)):
        super().__init__(prob)
        self.allowed_planes = allowed_planes
        self.rotations = rotations

    def augment(self, image, target, mask, wmaps):
        k = np.random.choice(self.rotations, len(self.allowed_planes))
        for i, axes in enumerate(self.allowed_planes):
            axes = np.random.choice(axes, 2, replace=False)  # direction of rotation
            image = np.rot90(image, k=k[i], axes=tuple(a + 1 for a in axes))
            target = np.rot90(target, k=k[i], axes=axes) if target is not None else None
            mask = np.rot90(mask, k=k[i], axes=axes) if mask is not None else None
            wmaps = np.rot90(wmaps, k=k[i], axes=axes) if wmaps is not None else None

        return image, target, mask, wmaps


class RandomPatchFlip(RandomAugmentation):
    def __init__(self, prob, allowed_axis):
        super().__init__(prob)
        self.allowed_axes = allowed_axis

    def augment(self, image, target, mask, wmaps):
        for axis in self.allowed_axes:
            image = np.flip(image, axis=axis + 1)
            target = np.flip(target, axis=axis) if target is not None else None
            mask = np.flip(mask, axis=axis) if mask is not None else None
            wmaps = np.flip(wmaps, axis=axis) if wmaps is not None else None

        return image, target, mask, wmaps


class RandomHistogramDeformation(RandomAugmentation):
    def __init__(self, prob, shift_std=0.05, scale_std=0.01, allow_mirror=False):
        super().__init__(prob)
        self.shift_std = shift_std
        self.scale_std = scale_std
        self.allow_mirror = allow_mirror

    def augment(self, image, target, mask, wmaps):
        num_channels = image.shape[0]
        shift = np.random.uniform(0, self.shift_std, num_channels)
        scale = np.random.normal(1, self.scale_std, num_channels)
        if self.allow_mirror:
            scale *= np.random.choice((-1, 1))
        image = (image.T * scale).T
        image = (image.T + shift).T
        return image, target, mask, wmaps


class RandomGammaCorrection(RandomAugmentation):
    def __init__(self, prob, gamma_std=.1):
        super().__init__(prob)
        self.gamma_std = gamma_std

    def augment(self, image, target, mask, wmaps):
        num_channels = image[0].shape[0]
        # gamma correction must be performed in the range of 0 to 1
        for i in range(len(image)):
            image_min = np.min(image[i])
            image_max = np.max(image[i])
            image[i] = (image[i] - image_min) / (image_max - image_min)
            gamma = np.random.normal(1, self.gamma_std, num_channels)
            image[i] = np.power(image[i].T, gamma).T
            image[i] = image[i] * (image_max - image_min) + image_min

        return image, target, mask, wmaps


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


class RandomGaussian(RandomAugmentation):
    def __init__(self, sigma, prob=0.5, random_sigma=False, sigma_min=0):
        super().__init__(prob)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.random_sigma = random_sigma

    def augment(self, image, target, mask, wmaps):
        if self.random_sigma:
            sigma = np.random.uniform(self.sigma_min, self.sigma)
        else:
            sigma = self.sigma

        image = gaussian_filter(image, sigma)

        return image, target, mask, wmaps


class RandomCropRescale(RandomAugmentation):
    def __init__(self, prob, max_crop, min_crop=1, centred=False, segmentation=False, percent=False,
                 uniform=True, order=3):
        super().__init__(prob)
        self.max_crop = max_crop
        self.min_crop = min_crop
        self.centred = centred
        self.segmentation = segmentation
        self.order = order
        self.percent = percent
        self.uniform = uniform

    def augment(self, image, target, mask, wmaps):
        ndim = target.ndim
        shape = np.array(target.shape)

        max_crop = to_array(self.max_crop, ndim)
        min_crop = to_array(self.min_crop, ndim)

        if self.uniform:
            crop_val = np.random.random() * (max_crop[0] - min_crop[0]) + min_crop[0]
            crop = to_array(crop_val, ndim)
        else:
            crop = [np.random.random() * (n_crop_max - n_crop_min) + n_crop_min
                    for n_crop_min, n_crop_max in zip(min_crop, max_crop)]

        if self.percent:
            crop = np.ceil([crop[i] * shape[i] for i in range(ndim)])

        crop = np.array([min(crop[i], shape[i] - 1) for i in range(ndim)], dtype=int)

        shape_crop = shape - crop
        scale = np.array(1. * (shape / shape_crop))

        if self.centred:
            crop_left = np.floor(crop / 2)
            crop_right = np.ceil(crop / 2)
        else:
            crop_left = [np.random.randint(n_crop) if n_crop > 0 else 0 for n_crop in crop]
            crop_right = crop - crop_left

        # get slc
        slc = [slice(None)] * len(shape)
        for axis in range(ndim):
            if crop[axis] == 0:
                slc[axis] = slice(shape[axis])
            else:
                slc[axis] = slice(crop_left[axis], -crop_right[axis])

        # crop & rescale
        image = np.stack([crop_rescale(channel, slc, scale, self.order) for channel in image])

        if self.segmentation:
            target = crop_rescale(target, slc, scale, self.order)
            mask = crop_rescale(mask, slc, scale, self.order)
            wmaps = crop_rescale(wmaps, slc, scale, self.order)

        return image, target, mask, wmaps


class RandomRotation(RandomAugmentation):
    def __init__(self, allowed_planes, max_angle, min_angle=0, prob=0.5,
                 cval=0, cval_target=0, cval_mask=0, order=3):
        super().__init__(prob)
        self.allowed_planes = allowed_planes
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.cval = cval
        self.cval_target = cval_target
        self.cval_mask = cval_mask
        self.order = order

    def augment(self, image, target, mask, wmaps):
        for i, axes in enumerate(self.allowed_planes):
            angle = np.random.choice([-1, 1]) * \
                    (np.random.random() * (self.max_angle - self.min_angle) + self.min_angle)
            for i in range(len(image)):
                image[i] = rotate(image[i], angle, axes=tuple(a + 1 for a in axes), reshape=False,
                                  cval=self.cval, order=self.order)
            target = rotate(target, angle, axes=axes, reshape=False,
                            cval=self.cval_target, order=self.order) if target is not None else None
            mask = rotate(mask, angle, axes=axes, reshape=False,
                          cval=self.cval_mask, order=self.order) if mask is not None else None
            wmaps = rotate(wmaps, angle, axes=axes, reshape=False,
                           cval=self.cval_mask, order=self.order) if mask is not None else None

        return image, target, mask, wmaps


class RandomInvert(RandomAugmentation):
    def __init__(self, prob=0.5):
        super().__init__(prob)

    def augment(self, image, target, mask, wmaps):
        if np.random.random_sample() > self.prob:
            return image, target, mask, wmaps

        image = -image

        return image, target, mask, wmaps

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
