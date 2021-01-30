# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division
import numpy as np
from math import ceil

def calc_rec_field_of_path_assuming_strides_1(kern_dims): # Used by modelParams.py to find default input-shape.
    # TODO: Remove
    if not kern_dims:  # list is []
        return 0
    
    n_dims = len(kern_dims[0])
    receptive_field = [1] * n_dims
    for dim_idx in range(n_dims):
        for layer_idx in range(len(kern_dims)):
            receptive_field[dim_idx] += kern_dims[layer_idx][dim_idx] - 1
    return receptive_field

    
def check_rec_field_vs_inp_dims(rec_field_dims, inp_dims):
    n_rf_dims = len(rec_field_dims)
    n_inp_dims = len(inp_dims)
    if n_rf_dims != n_inp_dims:
        print("ERROR: [in function check_rec_field_vs_inp_dims()] : Receptive field and image segment have different "
              "number of dimensions! (should be 3 for both! Exiting!)")
        exit(1)
    for dim_i in range(n_rf_dims):
        if rec_field_dims[dim_i] > inp_dims[dim_i]:
            print("ERROR: [in function check_rec_field_vs_inp_dims()] : The segment-size (input) should be at least "
                  "as big as the receptive field of the model! This was not found to hold! "
                  "Dimensions of Receptive Field:", rec_field_dims, ". Dimensions of Segment: ", inp_dims)
            return False
    return True


def check_kern_dims_per_l_correct_3d_and_n_layers(kern_dims_per_layer, n_layers):
    # kern_dims_per_layer : a list with sublists. One sublist per layer.
    # Each sublist should have 3 integers, specifying the dimensions of the kernel at the corresponding layer of
    # the pathway. eg: kern_dims_per_layer = [ [3,3,3], [3,3,3], [5,5,5] ]
    if kern_dims_per_layer is None or len(kern_dims_per_layer) != n_layers:
        return False
    for kern_dims in kern_dims_per_layer:
        if len(kern_dims) != 3:
            return False
    return True


def subsample_factor_is_even(subs_factor):
    for dim_i in range(len(subs_factor)):
        if subs_factor[dim_i] % 2 != 1:
            return False
    return True

