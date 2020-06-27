class FCKernDimFMLengthNotEqualException(Exception):
    def __init__(self, kern_dim_len, n_fm_len):
        self.kern_dim_len = kern_dim_len
        self.n_fm_len = n_fm_len
        self.message = "Need one Kernel-Dimensions per layer of FC path, equal to length of number-of-FMs-in-FC +1 (for classif layer)"

    def __str__(self):
        return "{} != {}; {}".format(self.kern_dim_len, self.n_fm_len + 1, self.message)


class NumClassInputException(Exception):
    pass


class NumChannelsInputException(Exception):
    pass


class NumFMNormInputException(Exception):
    pass


class KernelDimNormInputException(Exception):
    pass


class ModelCfgListOfListException(Exception):
    def __init__(self, data):
        msg = (
            'ERROR: variable "',
            data,
            '" given in modelConfig.cfg should be either a list of '
            "integers, or a list of lists of integers, in case multiple lower-scale pathways are wanted. "
            "Please correct it. Exiting.",
        )
        super().__init__(msg)


class NumFMSubsampledInputException(Exception):
    pass


class KernelDimSubsampledInputException(Exception):
    pass


class NormAndSubsampledReceptiveFieldNotEqualException(Exception):
    def __init__(self, rec_field_norm, rec_field_subs):
        msg = (
            "ERROR: The receptive field of the normal pathway was calculated = ",
            len(rec_field_norm),
            " while the receptive field of the subsampled pathway was calculated=",
            len(rec_field_subs),
            ". "
            "Because of limitations in current version, the two pathways must have the save size of receptive field. "
            'Please provide a combination of "numberFMsPerLayerSubsampled" and "kernelDimPerLayerSubsampled" '
            "that gives the same size of field as the normal pathway. If unsure of how to proceed, "
            'please ommit specifying "numberFMsPerLayerSubsampled" and "kernelDimPerLayerSubsampled" in '
            "the config file, and the second subsampled pathway will be automatically created to mirror the normal. "
            "Else, if you want to just specify the number of Feature Maps in the subsampled, "
            'provide "numberFMsPerLayerSubsampled" = [num-FMs-layer1, ..., num-FMs-layerN], with N the '
            "same number as the normal pathway, and we will then use the same kernel-sizes as the normal pathway. "
            "Exiting!",
        )
        super().__init__(msg)


class SubsampleFactorInputException(Exception):
    pass


class SegmentsDimInputException(Exception):
    pass


class ConvWInitTypeInputException(Exception):
    pass


class ActivationFunctionInputException(Exception):
    pass


class ResConnectionInputException(Exception):
    pass
