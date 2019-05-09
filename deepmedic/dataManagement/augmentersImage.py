# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage

class AugmenterAffineTransformation(object):
    def __init__(self, prob, max_rot_x, max_rot_y, max_rot_z, max_scaling, seed=64):
        self.prob = prob # Probability of applying the transformation.
        self.max_rot_x = max_rot_x
        self.max_rot_y = max_rot_y
        self.max_rot_z = max_rot_z
        self.max_scaling = max_scaling
        self.rng = np.random.RandomState(seed)

    def _get_random_transformation(self):
        theta_x = self.rng.uniform(-self.max_rot_x, self.max_rot_x) * np.pi / 180.
        rot_x = np.array([ [np.cos(theta_x), -np.sin(theta_x), 0.],
                           [np.sin(theta_x), np.cos(theta_x), 0.],
                           [0., 0., 1.]])
        
        theta_y = self.rng.uniform(-self.max_rot_y, self.max_rot_y) * np.pi / 180.
        rot_y = np.array([ [np.cos(theta_y), 0., np.sin(theta_y)],
                           [0., 1., 0.],
                           [-np.sin(theta_y), 0., np.cos(theta_y)]])
        
        theta_z = self.rng.uniform(-self.max_rot_z, self.max_rot_z) * np.pi / 180.
        rot_z = np.array([ [1., 0., 0.],
                           [0., np.cos(theta_z), -np.sin(theta_z)],
                           [0., np.sin(theta_z), np.cos(theta_z)]])
        
        # Sample the scale (zoom in/out)
        # TODO: Non isotropic?
        scale = np.eye(3, 3) * self.rng.uniform(1 - self.max_scaling, 1 + self.max_scaling)
        
        # Affine transformation matrix.
        transformation_mtx = np.dot( scale, np.dot(rot_z, np.dot(rot_x, rot_y)) )
        
        return transformation_mtx

    def _apply_transformation(self, image, transf_mtx, interp_order=3., boundary_mode='nearest', cval=0.):
        # image should be 3 dimensional (Height, Width, Depth). Not multi-channel.
        # interp_order: Integer. 3 for images, 0 for nearest neighbour on masks (GT & brainmasks)
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: float. value given to boundaries if mode is constant.
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
    
    def __call__(self, images_l, interp_orders, boundary_modes, cval=None):
        # images_l : List of images
        # intrp_orders : List of integers. Orders of bsplines for interpolation, one per image in images_l.
        #                Suggested: 3 for images. 1 is like linear. 0 for masks/labels, like NN.
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: single float value. Value given to boundaries if mode is constant.
        if self.rng.random_sample() > self.prob:
            return images_l

        transf_mtx = self._get_random_transformation()
        
        new_images_l = []
        for image, interp_order, boundary_mode in zip(images_l, interp_orders, boundary_modes):
            new_images_l.append( self._apply_transformation(image,
                                                            transf_mtx,
                                                            interp_order,
                                                            boundary_mode,
                                                            cval)
                                )
        return new_images_l


