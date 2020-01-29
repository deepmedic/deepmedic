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
