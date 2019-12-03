from deepmedic.gui.config_window import ConfigWindow
from deepmedic.gui.ui_preproc_config_create import UiPreprocConfig
from deepmedic.frontEnd.configParsing.preprocConfig import PreprocConfig
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
                                        'Correcting Orientation, Spacing, and Data Type...')
        self.resample_progress.hide()
        self.create_mask_progress = ProgressBar(self.ui.create_mask_progress, self.ui.create_mask_text,
                                                'Creating Masks...')
        self.create_mask_progress.hide()
        self.resize_progress = ProgressBar(self.ui.resize_progress, self.ui.resize_text,
                                           'Resizing Images...')
        self.resize_progress.hide()

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
        orientation_corr = self.findChild(QtWidgets.QCheckBox, 'preproc_orientation_checkbox').isChecked()
        resample_imgs = self.findChild(QtWidgets.QCheckBox, 'preproc_resample_checkbox').isChecked()
        spacing = self.get_text_value('preproc_pixelSpacing_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_pixelSpacing_lineedit'))
        change_pixel_type = self.findChild(QtWidgets.QCheckBox, 'preproc_changePixelType_checkbox').isChecked()
        pixel_type = self.get_text_value('preproc_pixelType_lineedit', self.findChild(QtWidgets.QComboBox, 'preproc_pixelType_combobox'))
        resize_imgs = self.findChild(QtWidgets.QCheckBox, 'preproc_resize_checkbox').isChecked()
        size = num(self.get_text_value('preproc_imgSize_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_imgSize_lineedit')))

        print(output_dir, orientation_corr, resample_imgs, spacing, change_pixel_type, pixel_type, resize_imgs, size)

        if os.path.isfile(csv):
            # check if Image is a column. Else throw error
            image_list = pd.read_csv(csv)
            image_list = image_list['Image']
        else:
            print('File not found')
            # Throw file not found error
            return

        if not resample_imgs:
            spacing = None

        # reorient and resample
        if orientation_corr or resample_imgs or change_pixel_type or resize_imgs:
            self.resample_progress.show()
            # Needs Data Type Conversion Code
            resample_image_list(image_list, orientation=orientation_corr,
                                params=ResampleParams(save_folder=output_dir,
                                                      spacing=spacing),
                                progress=self.resample_progress)
        # Get Mask

        # Resize
