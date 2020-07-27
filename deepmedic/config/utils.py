def calc_rec_field_of_path_assuming_strides_1(kern_dims):  # Used by modelParams.py to find default input-shape.
    # TODO: Remove
    if not kern_dims:  # list is []
        return 0

    n_dims = len(kern_dims[0])
    receptive_field = [1] * n_dims
    for dim_idx in range(n_dims):
        for layer_idx in range(len(kern_dims)):
            receptive_field[dim_idx] += kern_dims[layer_idx][dim_idx] - 1
    return receptive_field
