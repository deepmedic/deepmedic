from deepmedic.gui.config_window import ConfigWindow
from deepmedic.gui.ui_preproc_config_create import UiPreprocConfig
from deepmedic.frontEnd.configParsing.preprocConfig import PreprocConfig
from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti
from deepmedic.dataManagement.data_checks import run_checks, resample_image_list, ResampleParams
from deepmedic.gui.config_utils import *
import pandas as pd
import os

from PySide2 import QtWidgets, QtGui


class ProgressBar(object):
    def __init__(self, bar, label=None, label_text=None):
        self.bar = bar
        self.bar.setMinimum(0)
        self.bar.setValue(0)
        self.label = label
        self.set_text(label_text)

    def increase_value(self):
        self.bar.setValue(self.bar.value() + 1)
        QtGui.qApp.processEvents()

    def show(self):
        self.bar.show()
        if self.label:
            self.label.show()

    def hide(self):
        self.bar.hide()
        if self.label:
            self.label.hide()

    def set_text(self, label_text):
        if self.label:
            self.label.setText(label_text)


class PreprocConfigWindow(ConfigWindow):
    def __init__(self, parent=None):
        super(PreprocConfigWindow, self).__init__(PreprocConfig, 'Preprocess Data', parent,
                                                  UiConfigClass=UiPreprocConfig)

        self.ui.data_checks_button.clicked.connect(self.run_data_checks)
        self.ui.preprocess_button.clicked.connect(self.preprocess)
        self.data_checks_progress = ProgressBar(self.ui.data_checks_progress)
        self.data_checks_progress.hide()
        self.resample_progress = ProgressBar(self.ui.resample_progress, self.ui.resample_text,
                                             'Preprocessing data...')
        self.resample_progress.hide()

    def run_data_checks(self):
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        self.data_checks_progress.show()
        check_text = run_checks(csv, csv=True, pixs=True, dims=True, dtypes=True, dirs=True,
                                disable_tqdm=False, html=True, progress=self.data_checks_progress)
        self.ui.data_checks_text.setText(check_text)

    def preprocess(self):
        # Get parameters from forms
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        output_dir = self.get_text_value('preproc_outputDir_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_outputDir_lineedit'))
        image_extension = self.get_text_value('preproc_extension_combobox', self.findChild(QtWidgets.QComboBox, 'preproc_extension_combobox'))
        orientation_corr = self.findChild(QtWidgets.QCheckBox, 'preproc_orientation_checkbox').isChecked()
        resample_imgs = self.findChild(QtWidgets.QCheckBox, 'preproc_resample_checkbox').isChecked()
        spacing = self.get_text_value('preproc_pixelSpacing_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_pixelSpacing_lineedit'))
        change_pixel_type = self.findChild(QtWidgets.QCheckBox, 'preproc_changePixelType_checkbox').isChecked()
        pixel_type = self.get_text_value('preproc_pixelType_combobox', self.findChild(QtWidgets.QComboBox, 'preproc_pixelType_combobox'))
        create_mask = self.findChild(QtWidgets.QCheckBox, 'preproc_createMask_checkbox').isChecked()
        thresh_low = self.get_text_value('preproc_threshLow_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_threshLow_lineedit'))
        thresh_high = self.get_text_value('preproc_threshHigh_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_threshHigh_lineedit'))
        mask_dir = self.get_text_value('preproc_maskDirectory_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_maskDirectory_lineedit'))
        mask_suffix = self.get_text_value('preproc_maskSuffix_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_maskSuffix_lineedit'))
        mask_pixel_type = self.get_text_value('preproc_maskPixelType_combobox', self.findChild(QtWidgets.QComboBox, 'preproc_maskPixelType_combobox'))
        mask_extension = self.get_text_value('preproc_maskExtension_combobox', self.findChild(QtWidgets.QComboBox, 'preproc_maskExtension_combobox'))
        resize_imgs = self.findChild(QtWidgets.QCheckBox, 'preproc_resize_checkbox').isChecked()
        size = self.get_text_value('preproc_imgSize_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_imgSize_lineedit'))
        use_mask = self.findChild(QtWidgets.QCheckBox, 'preproc_useMask_checkbox').isChecked()
        use_centre_mass = self.findChild(QtWidgets.QCheckBox, 'preproc_centreMass_checkbox').isChecked()

        print(output_dir, orientation_corr, resample_imgs, spacing, change_pixel_type, pixel_type, resize_imgs, size)

        if os.path.isfile(csv):
            # check if Image is a column. Else throw error
            input_df = pd.read_csv(csv)
            image_list = input_df['Image']
            try:
                mask_list = input_df['Mask']
            except KeyError:
                mask_list = None
            try:
                target_list = input_df['Target']
            except KeyError:
                target_list = None
        else:
            print('File not found')
            # Throw file not found error
            return

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.resample_progress is not None:
            self.resample_progress.show()
            self.resample_progress.bar.setMaximum(len(image_list))

        suffix = ''
        if not suffix == '':
            suffix = '_' + suffix

        if not mask_suffix == '':
            mask_suffix = '_' + mask_suffix

        for i in range(len(image_list)):
            image_path = image_list[i]
            image = NiftiImage(image_path)
            if mask_list is not None:
                mask_path = mask_list[i]
                mask = NiftiImage(mask_path)
            else:
                mask_path = None
                mask = None
            if target_list is not None:
                target_path = target_list[i]
                target = NiftiImage(target_path)
            else:
                target_path = None
                target = None

            path_split = image_path.split('.')
            image_name = path_split[0]
            if output_dir:
                image_save_name = os.path.join(output_dir, image_name.split('/')[-1])
            else:
                image_save_name = image_name
            if not image_extension:
                image_extension = '.' + '.'.join(path_split[1:])

            # convert type
            if change_pixel_type:
                image.change_pixel_type(pixel_type)

            # reorient
            if orientation_corr:
                image.reorient()
                if mask:
                    mask.reorient()
                if target:
                    target.reorient()

            # resample (spacing)
            if resample_imgs:
                image.resample(spacing=spacing)
                if mask:
                    mask.resample(spacing=spacing)
                if target:
                    target.resample(spacing=spacing)

            # create mask
            if create_mask:
                mask = NiftiImage(image=image.get_mask(thresh_low, thresh_high))
                if mask_dir:
                    mask_save_name = os.path.join(mask_dir, image_name.split('/')[-1])
                else:
                    mask_save_name = image_save_name
                if not mask_extension:
                    mask_extension = image_extension
                if mask_pixel_type:
                    mask.change_pixel_type(mask_pixel_type)
            else:
                mask = None

            # resize
            if resize_imgs:
                centre_mass = use_centre_mass
                if mask:
                    crop_mask = True
                    if use_mask:
                        centre_mass = True
                else:
                    crop_mask = False
                image.resize(size, mask, centre_mass=centre_mass, crop_mask=crop_mask)
                mask = target.resize(size, mask, centre_mass=centre_mass, crop_mask=crop_mask)

            # save image
            if output_dir:
                save_nifti(image.open(), image_save_name + suffix + image_extension)
                if mask:
                    save_nifti(mask.open(), mask_save_name + mask_suffix + mask_extension)
                if target:
                    pass
                    # save_nifti(target.open(), mask_save_name + mask_suffix + mask_extension)

            if self.resample_progress is not None:
                self.resample_progress.increase_value()
