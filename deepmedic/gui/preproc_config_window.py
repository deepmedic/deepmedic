from deepmedic.gui.config_window import ConfigWindow
from deepmedic.gui.ui_preproc_config_create import UiPreprocConfig
from deepmedic.frontEnd.configParsing.preprocConfig import PreprocConfig
from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti
from deepmedic.dataManagement.data_checks import run_checks, resample_image_list, ResampleParams
from deepmedic.gui.config_utils import *
import pandas as pd
import os

from PySide2 import QtWidgets, QtGui


def get_save_name(image_path, output_dir, image_extension):
    path_split = image_path.split('.')
    image_name = path_split[0]
    if output_dir:
        image_save_name = os.path.join(output_dir, image_name.split('/')[-1])
    else:
        image_save_name = image_name
    if not image_extension:
        image_extension = '.' + '.'.join(path_split[1:])

    return image_name, image_save_name, image_extension


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

        self.dchecks_sug = None
        self.ui.suggested_button.clicked.connect(self.fill_in_sug)
        self.ui.suggested_button.hide()

    def run_data_checks(self):
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        self.data_checks_progress.show()
        check_text, self.dchecks_sug = run_checks(csv, csv=True,
                                                  pixs=True, dims=True, dtypes=True, dirs=True, sizes=True,
                                                  disable_tqdm=False, html=True, progress=self.data_checks_progress)
        self.ui.data_checks_text.setText(check_text)
        self.ui.suggested_button.show()

    def fill_in_sug(self):
        print(self.dchecks_sug)
        if self.dchecks_sug:
            if self.dchecks_sug['direction']:
                self.findChild(QtWidgets.QCheckBox, 'preproc_orientation_checkbox').setChecked(True)
            if self.dchecks_sug['spacing']:
                self.findChild(QtWidgets.QCheckBox, 'preproc_resample_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit,
                               'preproc_pixelSpacing_lineedit').setText(str(self.dchecks_sug['spacing']))
            if self.dchecks_sug['size']:
                self.findChild(QtWidgets.QCheckBox, 'preproc_resize_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit, 'preproc_imgSize_lineedit').\
                    setText(str(self.dchecks_sug['size']))
                combo = self.findChild(QtWidgets.QComboBox, 'preproc_imgSize_combobox')
                index = combo.findText('pixels', QtCore.Qt.MatchFixedString)
                if index < 0:
                    index = 1
                combo.setCurrentIndex(index)

            if self.dchecks_sug['dtype']:
                self.findChild(QtWidgets.QCheckBox, 'preproc_changePixelType_checkbox').setChecked(True)
                combo = self.findChild(QtWidgets.QComboBox, 'preproc_pixelType_combobox')
                index = combo.findText(self.dchecks_sug['dtype'], QtCore.Qt.MatchFixedString)
                if index >= 0:
                    combo.setCurrentIndex(index)

    def preprocess(self):
        # Get parameters from forms
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        output_dir = self.get_text_value('output_outputDir_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputDir_lineedit'))
        save_csv = self.findChild(QtWidgets.QCheckBox, 'output_saveCsv_checkbox').isChecked()
        out_csv_dir = self.get_text_value('output_outputCsvDir_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputCsvDir_lineedit'))
        out_csv_name = self.get_text_value('output_outputCsvName_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputCsvName_lineedit'))
        image_extension = self.get_text_value('output_extension_combobox', self.findChild(QtWidgets.QComboBox, 'output_extension_combobox'))
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
        size_units = self.get_text_value('preproc_imgSize_combobox', self.findChild(QtWidgets.QComboBox, 'preproc_imgSize_combobox'), elem_type='Units')
        use_mask = self.findChild(QtWidgets.QCheckBox, 'preproc_useMask_checkbox').isChecked()
        use_centre_mass = self.findChild(QtWidgets.QCheckBox, 'preproc_centreMass_checkbox').isChecked()

        print(output_dir, orientation_corr, resample_imgs, spacing, change_pixel_type, pixel_type, resize_imgs, size, size_units)

        if resize_imgs and spacing and size_units == 'mm':
            size = tuple([int(a * b) for a, b in zip(spacing, size)])

        if os.path.isfile(csv):
            # check if Image is a column. Else throw error
            image_list = pd.read_csv(csv)
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

        if not create_mask:
            mask_suffix = ''
        if not mask_suffix == '':
            mask_suffix = '_' + mask_suffix

        for i, row in image_list.iterrows():

            image_path = row['Image']
            try:
                mask_path = row['Mask']
            except KeyError:
                mask_path = None
            try:
                target_path = row['Target']
            except KeyError:
                target_path = None

            image = NiftiImage(image_path, mask_path, target_path)

            (image_name,
             image_save_name,
             image_extension) = get_save_name(image_path, output_dir, image_extension)

            if mask_path:
                (mask_name,
                 mask_save_name,
                 mask_extension) = get_save_name(mask_path, output_dir, None)

            if target_path:
                (target_name,
                 target_save_name,
                 target_extension) = get_save_name(target_path, output_dir, None)

            # convert type
            if change_pixel_type:
                image.change_pixel_type(pixel_type)

            # reorient
            if orientation_corr:
                image.reorient()

            # resample (spacing)
            if resample_imgs:
                image.resample(spacing=spacing)

            # create mask
            if create_mask:
                image.get_mask(thresh_low, thresh_high)
                if mask_dir:
                    mask_save_name = os.path.join(mask_dir, image_name.split('/')[-1])
                else:
                    mask_save_name = image_save_name
                if not mask_extension:
                    mask_extension = image_extension
                if mask_pixel_type:
                    image.mask.change_pixel_type(mask_pixel_type)

            # resize
            if resize_imgs:
                if not spacing and size_units == 'mm':
                    this_size = tuple([int(a * b) for a, b in zip(image.get_spacing(), size)])
                else:
                    this_size = size
                image.resize(this_size, centre_mass=use_centre_mass, use_mask=use_mask)

            # save image
            if output_dir:
                new_image_name = image_save_name + suffix + image_extension
                save_nifti(image.open(), new_image_name)
                image_list.at[i, 'Image'] = new_image_name
                if image.mask:
                    new_mask_name = mask_save_name + mask_suffix + mask_extension
                    save_nifti(image.mask.open(), new_mask_name)
                    image_list.at[i, 'Mask'] = new_mask_name
                if image.target:
                    new_target_name = target_save_name + target_extension
                    save_nifti(image.target.open(), new_target_name)
                    image_list.at[i, 'Target'] = new_target_name

            if self.resample_progress is not None:
                self.resample_progress.increase_value()

        if save_csv:
            if not out_csv_dir:
                out_csv_dir = os.path.dirname(csv)
            if not out_csv_name:
                out_csv_name = os.path.basename(csv)

            os.makedirs(out_csv_dir, exist_ok=True)
            image_list.to_csv(os.path.join(out_csv_dir, out_csv_name))
