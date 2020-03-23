# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

from deepmedic.frontEnd.configParsing.config import Config, ConfigData, ConfigElem


class PreprocConfig(Config):
    # define sections
    config_data = ConfigData()

    config_data.set_curr_section('data', "Input Data")
    # Optional but highly suggested.
    INPUT_CSV = \
        config_data.add_elem("inputCsv", elem_type='File', options='csv',
                             description='Data CSV', required=True,
                             info="CSV with the input data. Columns should be named as follows:\n"
                                  " - 'Channel_##'   path to images for channel ##\n"
                                  " - 'Mask' (optional)   path to sampling masks\n"
                                  " - 'Target' (optional)   path to segmentation targets.\n"
                                  " - 'Id' (optional)   subject ID")

    # ===Preprocessing Parameters===
    config_data.set_curr_section('output', "Output Parameters")

    # [REQUIRED] Output:
    OUTPUT_DIR = \
        config_data.add_elem("outputDir", elem_type='Folder', required=True,
                             description='Output Directory',
                             info="The main folder that the output will be placed in.")

    SAVE_CSV = \
        config_data.add_elem("saveCsv", description='Save Output CSV', elem_type='Bool',
                             info="Save new updated CSV file with preprocessed data paths.\n"
                                  "Requires at least one of the bottom two options.")

    OUTPUT_CSV_DIR = \
        config_data.add_elem("outputCsvDir",
                             description='     Save Directory', parent=SAVE_CSV, elem_type='Folder',
                             info="Directory in which to save the output CSV file.\n"
                                  "[Default] Will save the file in the same directory as the input CSV.")

    OUTPUT_CSV_NAME = \
        config_data.add_elem("outputCsvName",
                             description='     Filename', parent=SAVE_CSV,
                             info="Name of the output CSV file.\n"
                                  "[Default] By default the name of the input file is used.")

    EXTENSION = \
        config_data.add_elem("extension",
                             description='Image File Format', elem_type='String', widget_type='combobox',
                             options=[".nii", ".nii.gz"],
                             info="File format to save the output images in. "
                                  "Compressed nifti (.nii.gz) takes up less memory. "
                                  "Default is to replicate the type of the input images.")

    USE_BASE_DIR = \
        config_data.add_elem("useBaseDir", description='Replicate Folder Structure', elem_type='Bool',
                             info="Replicate the folder structure from the Base Directory in the Output folder. "
                                  "If not selected, the default structure is as follows:\n"
                                  "[output_dir]/[subj_id]/[channel]/[image_name].nii[.gz]")
    BASE_DIR = \
        config_data.add_elem("baseDir", elem_type='Folder', parent=USE_BASE_DIR,
                             description='     Data Base Directory',
                             info='Base Directory to use as reference for the creation'
                                         ' of the output folder structure.')

    # ===Preprocessing Parameters===
    config_data.set_curr_section('preproc', "Preprocessing Parameters")

    ORIENTATION = \
        config_data.add_elem("orientation", description='Orientation Normalisation', elem_type='Bool',
                             info="Reorient images to standard radiology view.")

    RESAMPLE = \
        config_data.add_elem("resample", description='Change Pixel Spacing', elem_type='Bool',
                             info="Resample images to a different pixel spacing, "
                                  "i.e. change the dimension of each pixel in the image.")

    SPACING = \
        config_data.add_elem("pixelSpacing",
                             description='     Pixel Spacing', parent=RESAMPLE,
                             info="The dimensions of each pixel. "
                                  "For most applications we recommend isotropic pixel spacing, e.g. (1.0, 1.0, 1.0)")

    RESIZE_INTRA = \
        config_data.add_elem("resizeIntraSubject", description='Resize Same Subject Images', elem_type='Bool',
                             info="Resize all images of the same subject to the dimensions of the reference column")

    RESIZE_INTRA_COLUMN = \
        config_data.add_elem("resizeIntraColumn", parent=RESIZE_INTRA,
                             description='     Reference Column', elem_type='String',
                             info="Reference column name to be used across the dataset to"
                                  " resize the same subject images. (e.g. 'Channel_1')")

    CHANGE_PIXEL_TYPE = \
        config_data.add_elem("changePixelType", description='Change Pixel Type', elem_type='Bool',
                             info="Save image with a certain pixel data type.")

    PIXEL_TYPE = \
        config_data.add_elem("pixelType", parent=CHANGE_PIXEL_TYPE,
                             description='     Pixel Type', elem_type='String', widget_type='combobox',
                             options=["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
                                      "float32", "float64"],
                             info="Data type the output images will be saved in "
                                  "(masks and targets, if provided, will retain the original format.)")

    RESIZE = \
        config_data.add_elem("resize", description='Resize All Images (Crop/Pad)', elem_type='Bool',
                             info="Resize images to uniform dimensions")

    IMAGE_SIZE =\
        config_data.add_elem("imgSize", description='     Image Size', parent=RESIZE, elem_type='Units',
                             options=['pixels', 'mm'],
                             info="The dimensions of the output image")

    CENTRE_MASS = \
        config_data.add_elem("centreMass", description='     Recentre on Centre of Mass', elem_type='Bool',
                             parent=RESIZE,
                             info="Recentre the cropped image around the centre of mass of the image. "
                                  "If masks are provided, the centre of mass of the mask will be used instead.")

    USE_MASK = \
        config_data.add_elem("useMask", description='     Recentre With Mask', elem_type='Bool', parent=RESIZE,
                             info="Use the masks to avoid cropping parts of the image with valuable information.")

    CREATE_MASK = \
        config_data.add_elem("createMask", description='Create Masks', elem_type='Bool',
                             info='Create masks using thresholds.')

    THRESH_LOW = \
        config_data.add_elem("threshLow", description='     Lower Threshold', parent=CREATE_MASK,
                             info="Low bound threshold. Leave empty if using only a high bound threshold.\n"
                                  "[Default: None]")

    THRESH_HIGH = \
        config_data.add_elem("threshHigh", description='     Higher Threshold', parent=CREATE_MASK,
                             info="High bound threshold. Leave empty if using only a low bound threshold.\n"
                                  "[Default: None]")

    MASK_DIR = \
        config_data.add_elem("maskDirectory", description='     Output Directory', parent=CREATE_MASK,
                             elem_type='Folder',
                             info="Directory in which to save the images.\n"
                                  "Masks will be saved in the same directory as the output images by default.")

    MASK_SUFFIX = \
        config_data.add_elem("maskSuffix", description='     Suffix', elem_type='String', parent=CREATE_MASK,
                             default='Mask',
                             info="What suffix to add to the image names. "
                                  "Images will be saved as [original name]_[suffix].[extension].")

    MASK_PIXEL_TYPE = \
        config_data.add_elem("maskPixelType", parent=CREATE_MASK,
                             description='     Pixel Type', elem_type='String', widget_type='combobox',
                             options=["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
                                      "float32", "float64"],
                             info="Data type the output masks will be saved in.")

    MASK_EXTENSION = \
        config_data.add_elem("maskExtension", parent=CREATE_MASK,
                             description='     File Format', elem_type='String', widget_type='combobox',
                             options=[".nii", ".nii.gz"],
                             info="File format to save the masks in. "
                                  "Default is to replicate the type of the input images.")

    THRESHOLD = \
        config_data.add_elem("thresh", description='Threshold Images', elem_type='Bool',
                             info='Cutoff Images using thresholds.')

    THRESH_LOW_CUT = \
        config_data.add_elem("threshLowCut", description='     Lower Threshold', parent=THRESHOLD,
                             info="Low bound threshold. Leave empty if using only a high bound threshold."
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)\n"
                                  "[Default: None]")

    THRESH_HIGH_CUT = \
        config_data.add_elem("threshHighCut", description='     Higher Threshold', parent=THRESHOLD,
                             info="High bound threshold. Leave empty if using only a low bound threshold."
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)\n"
                                  "[Default: None]")

    NORM_RANGE = \
        config_data.add_elem("normRange", description='Normalise to a Range', elem_type='Bool',
                             info='Normalise to a range')

    LOW_RANGE_ORIG = \
        config_data.add_elem("lowRangeOrig", description='     Original Low Bound', parent=NORM_RANGE,
                             info="Low bound threshold. Leave empty if using only a high bound threshold. "
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)\n"
                                  "[Default: None]")

    HIGH_RANGE_ORIG = \
        config_data.add_elem("highRangeOrig", description='     Original High Bound', parent=NORM_RANGE,
                             info="High bound threshold. Leave empty if using only a low bound threshold. "
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)\n"
                                  "[Default: None]")

    LOW_RANGE_TARGET = \
        config_data.add_elem("lowRangeTarget", description='     Preprocessed Low Bound', parent=NORM_RANGE,
                             info="Low bound threshold. Leave empty if using only a high bound threshold. "
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)",
                             default=1)

    HIGH_RANGE_TARGET = \
        config_data.add_elem("highRangeTarget", description='     Preprocessed High Bound', parent=NORM_RANGE,
                             info="High bound threshold. Leave empty if using only a low bound threshold. "
                                  "This can be a list (one threshold per channel) or a single value "
                                  "(which will be used for all channels)",
                             default=1)

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)
