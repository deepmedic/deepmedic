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
                             info="CSV with the input data. Columns should be as follows:\n"
                                  "'Image' - path to images\n"
                                  "'Mask' (optional) - path to sampling masks\n"
                                  "'Target' (optional) - path to segmentation targets.")

    # ===Preprocessing Parameters===
    config_data.set_curr_section('preproc', "Preprocessing Parameteres")

    # [REQUIRED] Output:
    OUTPUT_DIR = \
        config_data.add_elem("outputDir", elem_type='Folder', required=True,
                             description='Output Directory',
                             info="The main folder that the output will be placed in .")

    ORIENTATION = \
        config_data.add_elem("orientation", description='Orientation Correction', elem_type='Bool',
                             info="Reorient images to standard radiology view.")

    RESAMPLE = \
        config_data.add_elem("resample", description='Resample Images', elem_type='Bool',
                             info="Resample images to uniform scaling")

    SCALING = \
        config_data.add_elem("pixelScaling",
                             description='   Pixel Scaling', parent=RESAMPLE,
                             info="The dimensions of each pixel. "
                                  "For most applications we recommend isotropic pixel scaling, e.g. (1.0, 1.0, 1.0)")

    PIXEL_TYPE = \
        config_data.add_elem("pixelType", parent=RESAMPLE,
                             description='   Pixel Type', elem_type='String', widget_type='combobox',
                             options=["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
                                      "float32", "float64"],
                             info="Data type the output images will be saved in "
                                  "(masks and targets will retain the original format.)")

    RESIZE = \
        config_data.add_elem("resize", description='Resize Images', elem_type='Bool',
                             info="Resize images to uniform dimensions")

    IMAGE_SIZE =\
        config_data.add_elem("imgSize", description='   Image Size', parent=RESIZE,
                             info="The dimensions of the output image")
    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)
