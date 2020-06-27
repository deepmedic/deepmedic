from typing import List
import warnings

from deepmedic.config.input import InputModelConfig
from deepmedic.config.utils import calc_rec_field_of_path_assuming_strides_1
from deepmedic.controller.utils import (
    check_kern_dims_per_l_correct_3d_and_n_layers,
    subsample_factor_is_even,
    check_rec_field_vs_inp_dims,
)
from deepmedic.exceptions import (
    FCKernDimFMLengthNotEqualException,
    NumClassInputException,
    NumChannelsInputException,
    NumFMNormInputException,
    KernelDimNormInputException,
    ModelCfgListOfListException,
    NumFMSubsampledInputException,
    KernelDimSubsampledInputException,
    NormAndSubsampledReceptiveFieldNotEqualException,
    SubsampleFactorInputException,
    SegmentsDimInputException,
    ConvWInitTypeInputException,
    ActivationFunctionInputException,
    ResConnectionInputException,
)


def assert_input_model_config(input_config: InputModelConfig):
    assert_n_classes(input_config.n_classes)
    assert_n_input_channels(input_config.n_input_channels)
    assert_n_fms_per_layer_normal(input_config.n_fm_norm)
    assert_kernel_dim_per_layer_normal(input_config.kern_dim_norm, len(input_config.n_fm_norm))
    assert_n_fms_per_subsampled_layer(input_config.n_fm_subs)
    n_layers_subs = len(input_config.n_fm_subs[0])
    if input_config.kern_dim_subs is None and n_layers_subs == len(input_config.n_fm_norm):
        kern_dims_per_l_subs = input_config.kern_dim_norm
    else:
        kern_dims_per_l_subs = input_config.kern_dim_subs
    assert_kernel_dims_per_layer_subsampled(
        kern_dims_per_l_subs, n_layers_subs, input_config.n_fm_subs, input_config.n_fm_norm, input_config.kern_dim_norm
    )
    if input_config.subs_factor is not None:
        assert_subsample_factors(input_config.subs_factor)
    if input_config.kern_dim_fc is not None and input_config.n_fm_fc is not None:
        if len(input_config.kern_dim_fc) != len(input_config.n_fm_fc) + 1:
            raise FCKernDimFMLengthNotEqualException(len(input_config.kern_dim_fc), len(input_config.n_fm_fc))

    if input_config.resid_conn_layers_norm is not None:
        assert_res_connections(input_config.resid_conn_layers_norm, "Normal")
    if input_config.resid_conn_layers_subs is not None:
        assert_res_connections(input_config.resid_conn_layers_subs, "Subsampled")
    if input_config.resid_conn_layers_fc is not None:
        assert_res_connections(input_config.resid_conn_layers_fc, "Fully Connected")
    rec_field_norm = calc_rec_field_of_path_assuming_strides_1(input_config.kern_dim_norm)
    inp_dims_hr_path = {"train": None, "val": None, "test": None}
    inp_dims_hr_path["train"] = input_config.seg_dim_train
    inp_dims_hr_path["val"] = input_config.seg_dim_val if input_config.seg_dim_val is not None else rec_field_norm
    inp_dims_hr_path["test"] = (
        input_config.seg_dim_infer if input_config.seg_dim_infer is not None else input_config.seg_dim_train
    )
    assert_inp_dims_hr_path(inp_dims_hr_path, rec_field_norm)
    if input_config.conv_w_init is not None:
        assert_conv_w_init_type(input_config.conv_w_init)
    if input_config.activ_func is not None:
        assert_activation_func(input_config.activ_func)


def warn_for_same_receptive_field():
    warnings.warn(
        "WARN: Because of limitations in the current version, the two pathways must have the same "
        "size of receptive field. If unsure of how to proceed, "
        'please ommit specifying "numberFMsPerLayerSubsampled" and "kernelDimPerLayerSubsampled" in '
        "the config file, and the second subsampled pathway will be automatically created to mirror the normal. "
        "Else, if you want to just specify the number of Feature Maps in the subsampled,"
        ' provide "numberFMsPerLayerSubsampled" = [num-FMs-layer1, ..., num-FMs-layerN], '
        "with N the same number as the normal pathway, and we will then use the same kernel-sizes as "
        "the normal pathway."
    )


def warn_sub_factor_odd():
    warnings.warn(
        "WARN: The system was only thoroughly tested for odd subsampling factor! Eg subsample_factors = [3,3,3]."
    )


def assert_n_classes(n_classes: int):
    if n_classes is not None:
        return True
    else:
        raise NumClassInputException(
            "ERROR: Number of classses not specified in the config file, which is required. "
            "Please specify in the format: numberOfOutputClasses = 3 (any integer). This number should be including "
            "the background class! For instance if the class is binary, set this to 2! Exiting!"
        )


def assert_n_input_channels(n_input_channels: int):
    if n_input_channels is not None:
        if n_input_channels <= 0:
            raise NumChannelsInputException(
                "AssertError {} > 0; Number of input channels should be greater than 0.".format(n_input_channels)
            )
        return True
    else:
        raise NumChannelsInputException(
            'ERROR: Parameter "numberOfInputChannels" not specified or specified smaller than 1. '
            "Please specify the number of input channels that will be used as input to the CNN, "
            "in the format: numberOfInputChannels = number (an integer > 0). Exiting!"
        )


def assert_n_fms_per_layer_normal(n_fms: List[int]):
    if n_fms is not None and len(n_fms) > 0:
        return True
    else:
        raise NumFMNormInputException(
            'ERROR: The required parameter "numberFMsPerLayerNormal" was either not given, or given an empty list. '
            "This parameter should be given in the format: "
            "numberFMsPerLayerNormal = [number-of-FMs-layer1, ..., number-of-FMs-layer-N], "
            "where each number is an integer greater than zero. "
            "It specifies the number of layers (specified by the number of entries in the list) "
            "and the number of Feature Maps at each layer of the normal-scale pathway. "
            "Please provide and retry. Exiting!"
        )


def assert_kernel_dim_per_layer_normal(kernel_dim_per_layer_normal: List[List[int]], n_layers_norm: int):
    if check_kern_dims_per_l_correct_3d_and_n_layers((kernel_dim_per_layer_normal), n_layers_norm):
        return True
    else:
        raise KernelDimNormInputException(
            'ERROR: The required parameter "kernelDimPerLayerNormal" was not provided, or provided incorrectly. '
            "It should be provided in the format: "
            "kernelDimPerLayerNormal = [ [dim1-kernels-layer-1, dim2-kernels-layer-1, dim3-kernels-layer-1], ..., "
            " ... [dim1-of-kernels-in-layer-N, dim2-of-kernels-in-layer-N, dim3-of-kernels-in-layer-N] ]. "
            "It is a list of sublists. One sublist should be provided per layer of the Normal pathway. "
            'Thus it should have as many entries as the entries in parameter "numberFMsPerLayerNormal". '
            "Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the "
            "dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for "
            "a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. "
            "Please fix and retry \n WARN: The kernel dimensions should be ODD-NUMBERS. "
            "System was not thoroughly tested for kernels of even dimensions! Exiting!"
        )


def _assert_list_of_list(data: List):
    if not isinstance(data, list):
        raise ModelCfgListOfListException(data)
    all_elements_are_lists = True
    no_element_is_list = True
    for element in data:
        if isinstance(element, list):
            no_element_is_list = False
        else:
            all_elements_are_lists = False
    if not (all_elements_are_lists or no_element_is_list):  # some are lists and some are not
        raise ModelCfgListOfListException(data)


def assert_n_fms_per_subsampled_layer(n_fms_per_subsampled_layer: List):
    # TODO: exist duplication with outside
    _assert_list_of_list(n_fms_per_subsampled_layer)
    if len(n_fms_per_subsampled_layer) == 0:
        return True
    len_of_first = len(n_fms_per_subsampled_layer[0])
    for sublist_i in range(len(n_fms_per_subsampled_layer)):
        if len_of_first != len(n_fms_per_subsampled_layer[sublist_i]):
            raise NumFMSubsampledInputException(
                'ERROR: Parameter "numberFMsPerLayerSubsampled" has been given as a list of sublists of integers. '
                "This triggers the construction of multiple low-scale pathways."
                "\tHowever currently this functionality requires that same number of layers is used in both pathways."
                '\tUser specified in "numberFMsPerLayerSubsampled" sublists of different length. '
                "Each list should have the same lenght, as many as the wanted number of layers. Please adress this. \nExiting."
            )
    return True


def assert_kernel_dims_per_layer_subsampled(
    kern_dims_per_l_subs, n_layers_subs, n_fms_per_l_subs, n_fms_norm, kern_dims_norm
):
    rec_field_norm = calc_rec_field_of_path_assuming_strides_1(kern_dims_norm)

    if kern_dims_per_l_subs is None and n_layers_subs != len(n_fms_norm):
        warn_for_same_receptive_field()
        raise KernelDimSubsampledInputException(
            "ERROR: It was requested to use the 2-scale architecture, with a subsampled pathway. "
            "Because of limitations in current version, the two pathways must have the save size of receptive field. "
            'By default, if "useSubsampledPathway" = True, '
            'and parameters "numberFMsPerLayerSubsampled" and "kernelDimPerLayerSubsampled" are not specified, '
            "the second pathway will be constructed symmetrical to the first. "
            'However, in this case, "numberFMsPerLayerSubsampled" was specified. '
            "It was found to have ",
            len(n_fms_per_l_subs),
            " entries, which specified this amount of layers in "
            "the subsampled pathway. This is different than the number of layers in the Normal pathway, "
            "specified to be: ",
            len(n_fms_norm),
            ". "
            'In this case, we require you to also provide the parameter "numberFMsPerLayerSubsampled", '
            "specifying kernel dimensions in the subsampled pathway, in a fashion that results in same size "
            "of receptive field as the normal pathway. \nExiting!",
        )

    # KERN_DIM_SUBS was specified. Now it's going to be tricky to make sure everything alright.
    elif not check_kern_dims_per_l_correct_3d_and_n_layers(kern_dims_per_l_subs, n_layers_subs):
        warn_for_same_receptive_field()
        raise KernelDimSubsampledInputException(
            'ERROR: The parameter "kernelDimPerLayerSubsampled" was not provided, or provided incorrectly. '
            "It should be provided in the format: "
            "kernelDimPerLayerSubsampled = [ [dim1-kernels-layer-1, dim2-kernels-layer-1, dim3-kernels-layer-1], ... "
            "... [dim1-kernels-layer-N, dim2-kernels-layer-N, dim3-kernels-layer-N] ]. "
            "It is a list of sublists. One sublist should be provided per layer of the SUBSAMPLED pathway. "
            'Thus it should have as many entries as the entries in parameter "numberFMsPerLayerSubsampled". '
            "(WARN: if the latter is not provided, it is by default taken equal to "
            'what specified for "numberFMsPerLayerNormal", in order to make the pathways symmetrical). '
            "Each sublist should contain 3 integer ODD numbers greater than zero, which should specify the "
            "dimensions of the 3-dimensional kernels. For instace: kernelDimPerLayerNormal = [[5,5,5],[3,3,3]] for "
            "a pathway with 2 layers, the first of which has 5x5x5 kernels and the second 3x3x3 kernels. "
            "Please fix and retry. (WARN: The kernel dimensions should be ODD-NUMBERS. "
            "System was not thoroughly tested for kernels of even dimensions!)\nExiting!"
        )
    else:  # kernel dimensions specified and are correct. Check the two receptive fields and ensure correctness.
        rec_field_subs = calc_rec_field_of_path_assuming_strides_1(kern_dims_per_l_subs)
        if rec_field_norm != rec_field_subs:
            raise NormAndSubsampledReceptiveFieldNotEqualException(rec_field_norm, rec_field_subs)
        # Everything alright, finally. Proceed safely...
    return True


def assert_subsample_factors(subsample_factors):
    if all(not isinstance(e, list) for e in subsample_factors):
        subsample_factors = [subsample_factors]
    _assert_list_of_list(subsample_factors)

    n_subs_paths = len(subsample_factors)
    for subs_path_i in range(n_subs_paths):
        if len(subsample_factors[subs_path_i]) != 3:
            warn_sub_factor_odd()
            raise SubsampleFactorInputException(
                'ERROR: The parameter "subsample_factors" must have 3 entries, one for each of the 3 dimensions. '
                "Please provide it in the format: subsample_factors = [subFactor-dim1, subFactor-dim2, subFactor-dim3]. "
                "Each of the entries should be an integer, eg [3, 3, 3].\nExiting!"
            )
        if not subsample_factor_is_even(subsample_factors[subs_path_i]):
            warn_sub_factor_odd()


def assert_inp_dims_hr_path(inp_dims_hr_path, rec_field_norm):
    if inp_dims_hr_path["train"] is None:
        raise SegmentsDimInputException(
            'ERROR: The parameter "segmentsDimTrain" was is required but not given. '
            "It specifies the size of the 3D segment that is given as input to the network. "
            "It should be at least as large as the receptive field of the network in each dimension. "
            "Please specify it in the format: segmentsDimTrain = [dim-1, dim-2, dim-3]. Exiting!"
        )
    for train_val_test in ["train", "val", "test"]:
        if len(rec_field_norm) != len(inp_dims_hr_path[train_val_test]):
            raise SegmentsDimInputException(
                "ERROR: Receptive field and image segment have different number of dimensions!"
                " (should be 3 for both! Exiting!)"
            )
        for rec_field, inp_dim in zip(rec_field_norm, inp_dims_hr_path[train_val_test]):
            if rec_field > inp_dim:
                raise SegmentsDimInputException(
                    "ERROR: The segment-size (input) should be at least as big as the receptive field of the model! "
                    "The network was made with a receptive field of dimensions: {}"
                    ". But in the case of: [{}] the dimensions of the input segment were specified smaller: {}. "
                    "Please fix this by adjusting number of layer and kernel dimensions! Exiting!".format(rec_field_norm, train_val_test, inp_dims_hr_path),
                )


def assert_conv_w_init_type(conv_w_init_type):
    if not conv_w_init_type[0] in ["normal", "fanIn"]:
        raise ConvWInitTypeInputException('ERROR: Parameter "convWeightsInit" has been given invalid value. Exiting!')


def assert_activation_func(activation_func):
    if not activation_func in ["linear", "relu", "prelu", "elu", "selu"]:
        raise ActivationFunctionInputException(
            'ERROR: Parameter "activ_function" has been given invalid value. Exiting!'
        )


def assert_res_connections(res_conn_at_layers, pathway_type: str):
    error_msg = (
        'ERROR: The parameter "layersWithResidualConn" for the [{pathway_type}] pathway was '
        "specified to include the number 1, ie the 1st layer.\n"
        "\t This is not an acceptable value, as a residual connection is made between the output of "
        "the specified layer and the input of the previous layer. There is no layer before the 1st!\n"
        "\t Provide a list that does not iinclude the first layer, eg layersWithResidualConnNormal = [4,6,8], "
        "or an empty list [] for no such connections. Exiting!"
    )
    if 1 in res_conn_at_layers:
        raise ResConnectionInputException(error_msg.format(pathway_type=pathway_type))
