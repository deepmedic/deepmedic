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

    config_data.set_curr_section('data', "Data")
    # Optional but highly suggested.
    INPUT_CSV = \
        config_data.add_elem("inputCsv", elem_type='File', options='csv',
                             description='Data CSV', required=True,
                             info="CSV with the input data. Columns should be named as follows:\n"
                                  " - 'Image'   path to images\n"
                                  " - 'Mask' (optional)   path to sampling masks\n"
                                  " - 'Target' (optional)   path to segmentation targets.")

    # ===Preprocessing Parameters===
    config_data.set_curr_section('preproc', "Preprocessing Parameters")

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
                             description='     Save Directory', parent=SAVE_CSV,
                             info="Directory in which to save the output CSV file.\n"
                                  "[Default] Will save the file in the same directory as the input CSV.")

    OUTPUT_CSV_NAME = \
        config_data.add_elem("outputCsvName",
                             description='     Filename', parent=SAVE_CSV,
                             info="Name of the output CSV file.\n"
                                  "[Default] By default the name of the input file is used.")

    EXTENSION = \
        config_data.add_elem("extension",
                             description='File Format', elem_type='String', widget_type='combobox',
                             options=[".nii", ".nii.gz"],
                             info="File format to save the output images in. "
                                  "Compressed nifti (.nii.gz) takes up less memory. "
                                  "Default is to replicate the type of the input images.")

    ORIENTATION = \
        config_data.add_elem("orientation", description='Orientation Correction', elem_type='Bool',
                             info="Reorient images to standard radiology view.")

    RESAMPLE = \
        config_data.add_elem("resample", description='Change Pixel Spacing', elem_type='Bool',
                             info="Resample images to uniform scaling")

    SPACING = \
        config_data.add_elem("pixelSpacing",
                             description='     Pixel Spacing', parent=RESAMPLE,
                             info="The dimensions of each pixel. "
                                  "For most applications we recommend isotropic pixel spacing, e.g. (1.0, 1.0, 1.0)")

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

    RESIZE = \
        config_data.add_elem("resize", description='Resize Images', elem_type='Bool',
                             info="Resize images to uniform dimensions")

    IMAGE_SIZE =\
        config_data.add_elem("imgSize", description='     Image Size', parent=RESIZE,
                             info="The dimensions of the output image")

    CENTRE_MASS = \
        config_data.add_elem("centreMass", description='     Recentre on Centre of Mass', elem_type='Bool',
                             parent=RESIZE,
                             info="Recentre the cropped image around the centre of mass of the image. "
                                  "If masks are provided, the centre of mass of the mask will be used instead.")

    USE_MASK = \
        config_data.add_elem("useMask", description='     Recentre With Mask', elem_type='Bool', parent=RESIZE,
                             info="Use the masks to avoid cropping parts of the image with valuable information.")

    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)
