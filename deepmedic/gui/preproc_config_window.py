from deepmedic.gui.config_window import ConfigWindow
from deepmedic.gui.ui_preproc_config_create import UiPreprocConfig
from deepmedic.frontEnd.configParsing.preprocConfig import PreprocConfig
from deepmedic.dataManagement.nifti_image import NiftiImage, save_nifti
from deepmedic.dataManagement.data_checks import run_checks, get_html_colour, get_bold_text, text_to_html
import pandas as pd
import os

from PySide2 import QtWidgets, QtGui, QtCore

TARGET = 'ground_truth'
MASK = 'roi_mask'
CHANNEL = 'channel'


def add_sufix(name, suffix):
    # fname, extension = os.path.splitext(name)
    split_text = name.split('.')
    fname = split_text[0]
    extension = '.' + '.'.join(split_text[1:])
    if suffix is None:
        suffix = ''
    return fname + suffix + extension


def get_save_name(image_path, output_dir, image_extension, base_dir=None, subj_id=None, channel_name=None):
    path_split = image_path.split('.')
    image_name = path_split[0]
    image_save_name = None
    if base_dir:
        if image_name.startswith(base_dir):
            image_save_name = os.path.join(output_dir, image_name[len(base_dir):].split('/')[-1])
        else:
            print('Image does not start with base_dir')
            exit(23)
    else:
        if subj_id is not None and channel_name is not None:
            image_save_name = os.path.join(output_dir, subj_id, channel_name, os.path.basename(image_name))
        else:
            print('subj_id and channel name need to be given')
            exit(24)
    if not image_extension:
        image_extension = '.' + '.'.join(path_split[1:])

    return image_save_name + image_extension


def get_channel_names(data_columns):
    channels = []
    for col in data_columns:
        if col.startswith(CHANNEL + '_') or col == 'Image':
            channels += [col]
    if not channels:
        channels = None
    return channels


def get_image_paths(row, channels):
    channel_paths = {}
    print(channels)
    for channel in channels:
        channel_paths[channel] = row[channel]

    try:
        mask_path = row[MASK]
    except KeyError:
        mask_path = None

    try:
        target_path = row[TARGET]
    except KeyError:
        target_path = None

    return channel_paths, mask_path, target_path


class ProgressBar(object):
    def __init__(self, bar, label=None, label_text=None):
        self.bar = bar
        self.bar.setMinimum(0)
        self.bar.setValue(0)
        self.label = label
        self.set_text(label_text)

    def increase_value(self):
        self.bar.setValue(self.bar.value() + 1)
        QtCore.QCoreApplication.processEvents()

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
        self.progress_text = 'Preprocessing data...'
        self.resample_progress = ProgressBar(self.ui.resample_progress, self.ui.resample_text,
                                             self.progress_text)
        self.resample_progress.hide()

        self.dchecks_sug = None
        self.data_cols = None
        self.ui.suggested_button.clicked.connect(self.fill_in_sug)
        self.ui.suggested_button.hide()

        self.fill_in_defaults()

    def reset_progress_text(self):
        self.resample_progress.hide()
        self.resample_progress.set_text(self.progress_text)

    def fill_in_defaults(self):
        self.findChild(QtWidgets.QCheckBox, 'output_saveCsv_checkbox').setChecked(True)

    def run_data_checks(self):
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        self.data_checks_progress.show()
        check_text, self.dchecks_sug = run_checks(csv, csv=True,
                                                  pixs=True, dims=True, dtypes=True, dirs=True, sizes=True,
                                                  intra_dims=True, intra_sizes=True,
                                                  disable_tqdm=False, html=True, progress=self.data_checks_progress)
        self.ui.data_checks_text.setText(check_text)
        self.ui.suggested_button.show()

    def fill_in_sug(self):
        print(self.dchecks_sug)
        if self.dchecks_sug:
            if self.dchecks_sug['direction'] is not None:
                self.findChild(QtWidgets.QCheckBox, 'preproc_orientation_checkbox').setChecked(True)
            if self.dchecks_sug['spacing'] is not None:
                self.findChild(QtWidgets.QCheckBox, 'preproc_resample_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit,
                               'preproc_pixelSpacing_lineedit').setText(str(self.dchecks_sug['spacing']))
            if self.dchecks_sug['intra_size']:
                self.findChild(QtWidgets.QCheckBox, 'preproc_resizeIntraSubject_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit, 'preproc_resizeIntraColumn_lineedit'). \
                    setText(self.dchecks_sug['intra_size'])
            if self.dchecks_sug['size'] is not None:
                self.findChild(QtWidgets.QCheckBox, 'preproc_resize_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit, 'preproc_imgSize_lineedit').\
                    setText(str(self.dchecks_sug['size']))
                combo = self.findChild(QtWidgets.QComboBox, 'preproc_imgSize_combobox')
                index = combo.findText('pixels', QtCore.Qt.MatchFixedString)
                if index < 0:
                    index = 1
                combo.setCurrentIndex(index)

            if self.dchecks_sug['dtype'] is not None:
                self.findChild(QtWidgets.QCheckBox, 'preproc_changePixelType_checkbox').setChecked(True)
                combo = self.findChild(QtWidgets.QComboBox, 'preproc_pixelType_combobox')
                index = combo.findText(self.dchecks_sug['dtype'], QtCore.Qt.MatchFixedString)
                if index >= 0:
                    combo.setCurrentIndex(index)

            if self.dchecks_sug['base_dir'] is not None and not self.dchecks_sug['base_dir'] == '':
                # self.findChild(QtWidgets.QCheckBox, 'output_useBaseDir_checkbox').setChecked(True)
                self.findChild(QtWidgets.QLineEdit,
                               'output_baseDir_lineedit').setText(str(self.dchecks_sug['base_dir']))

    def preprocess(self):

        # Get parameters from forms
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        output_dir = self.get_text_value('output_outputDir_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputDir_lineedit'))
        save_csv = self.findChild(QtWidgets.QCheckBox, 'output_saveCsv_checkbox').isChecked()
        out_csv_dir = self.get_text_value('output_outputCsvDir_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputCsvDir_lineedit'))
        out_csv_name = self.get_text_value('output_outputCsvName_lineedit', self.findChild(QtWidgets.QLineEdit, 'output_outputCsvName_lineedit'))
        image_extension = self.get_text_value('output_extension_combobox', self.findChild(QtWidgets.QComboBox, 'output_extension_combobox'))
        use_base_dir = self.findChild(QtWidgets.QCheckBox, 'output_useBaseDir_checkbox').isChecked()
        if use_base_dir:
            base_dir = self.findChild(QtWidgets.QLineEdit, 'output_baseDir_lineedit').text()
        else:
            base_dir = None
        orientation_corr = self.findChild(QtWidgets.QCheckBox, 'preproc_orientation_checkbox').isChecked()
        resample_imgs = self.findChild(QtWidgets.QCheckBox, 'preproc_resample_checkbox').isChecked()
        spacing = self.get_text_value('preproc_pixelSpacing_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_pixelSpacing_lineedit'))
        resize_intra = self.findChild(QtWidgets.QCheckBox, 'preproc_resizeIntraSubject_checkbox').isChecked()
        resize_intra_col = self.get_text_value('preproc_resizeIntraColumn_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_resizeIntraColumn_lineedit'))
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
        threshold = self.findChild(QtWidgets.QCheckBox, 'preproc_thresh_checkbox').isChecked()
        thresh_low_cut = self.get_text_value('preproc_threshLowCut_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_threshLowCut_lineedit'))
        thresh_high_cut = self.get_text_value('preproc_threshHighCut_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_threshHighCut_lineedit'))
        norm_range = self.findChild(QtWidgets.QCheckBox, 'preproc_normRange_checkbox').isChecked()
        low_range_orig = self.get_text_value('preproc_lowRangeOrig_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_lowRangeOrig_lineedit'))
        high_range_orig = self.get_text_value('preproc_highRangeOrig_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_highRangeOrig_lineedit'))
        low_range_target = self.get_text_value('preproc_lowRangeTarget_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_lowRangeTarget_lineedit'))
        high_range_target = self.get_text_value('preproc_highRangeTarget_lineedit', self.findChild(QtWidgets.QLineEdit, 'preproc_highRangeTarget_lineedit'))

        self.reset_progress_text()

        # check output
        if not output_dir:
            output_dir_label = self.findChild(QtWidgets.QLabel, 'output_outputDir_label')
            text = output_dir_label.text()
            output_dir_label.setText(get_html_colour(text, colour='red'))

            current_text = self.resample_progress.label.text()
            new_text = 'ERROR: No ' + get_bold_text('Output Directory') + ' was given.'
            current_text += '\n' + get_html_colour(new_text, colour='red')
            self.resample_progress.set_text(text_to_html(current_text))
            self.resample_progress.show()

        # # check csv
        # if save_csv:
        #     if not (out_csv_dir or out_csv_dir):
        #         text = self.ui.output_saveCsv_checkbox.text()
        #         self.ui.output_saveCsv_checkbox.setText()

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
        if not (mask_suffix == '' or mask_suffix is None):
            mask_suffix = '_' + mask_suffix

        channel_names = get_channel_names(image_list.columns)
        print('names: ' + str(channel_names))

        for i, row in image_list.iterrows():

            (channel_paths,
             mask_path,
             target_path) = get_image_paths(row, channel_names)

            try:
                subj_id = str(row['case_index'])
            except KeyError:
                subj_id = str(i)

            print(channel_paths)

            image = NiftiImage([channel_paths[n] for n in channel_names], mask_path, target_path, channel_names=channel_names)

            channel_save_names = {}
            for channel in channel_names:
                channel_path = channel_paths[channel]
                image_save_name = get_save_name(channel_path, output_dir, image_extension,
                                                base_dir=base_dir, subj_id=subj_id, channel_name=channel)
                channel_save_names[channel] = image_save_name

            if mask_path:
                mask_save_name = get_save_name(mask_path, output_dir, None,
                                               base_dir=base_dir, subj_id=subj_id, channel_name=MASK)

            if target_path:
                target_save_name = get_save_name(target_path, output_dir, None,
                                                 base_dir=base_dir, subj_id=subj_id, channel_name=TARGET)

            # convert type
            if change_pixel_type:
                image.change_pixel_type(pixel_type)

            # reorient
            if orientation_corr:
                image.reorient()

            # resample (spacing)
            if resample_imgs:
                image.resample(spacing=spacing)

            # resize
            if resize_imgs:
                if not spacing and size_units == 'mm':
                    this_size = tuple([int(a * b) for a, b in zip(image.get_spacing(), size)])
                else:
                    this_size = size
                image.resize(this_size, centre_mass=use_centre_mass, use_mask=use_mask)

            # resize (intra-subject)
            if resize_intra:
                image.resample(ref_image=image.channels[resize_intra_col])

            # create  <--------------------------------------------------------- R E V I E W --------------------
            if create_mask:
                image.get_mask(thresh_low, thresh_high)
                if not use_base_dir or not (mask_dir or mask_suffix):
                    mask_save_name = get_save_name(channel_save_names[channel_names[0]], output_dir, None,
                                                   base_dir=None, subj_id=subj_id, channel_name=MASK)
                elif mask_dir:
                    mask_save_name = os.path.join(mask_dir, channel_save_names[channel_names[0]].split('/')[-1])
                elif mask_suffix:
                    mask_save_name = channel_save_names[channel_names[0]]

                if not mask_extension:
                    mask_extension = image_extension
                if mask_pixel_type:
                    image.mask.change_pixel_type(mask_pixel_type)

            # cutoff
            if threshold:
                image.thresh_cutoff(thresh_low_cut, thresh_high_cut)

            # normalise range
            if norm_range:
                image.norm_range(low_range_orig, high_range_orig, low_range_target, high_range_target)

            # save image
            if output_dir:
                for channel in channel_names:
                    new_image_name = add_sufix(channel_save_names[channel], suffix)
                    print(new_image_name)
                    save_nifti(image.channels[channel].open(), new_image_name)
                    image_list.at[i, channel] = new_image_name

                if image.mask:
                    new_mask_name = add_sufix(mask_save_name, mask_suffix)
                    save_nifti(image.mask.open(), new_mask_name)
                    image_list.at[i, MASK] = new_mask_name
                if image.target:
                    new_target_name = target_save_name
                    save_nifti(image.target.open(), new_target_name)
                    image_list.at[i, TARGET] = new_target_name

            if self.resample_progress is not None:
                self.resample_progress.increase_value()

        if save_csv:
            if not out_csv_dir:
                out_csv_dir = os.path.dirname(csv)
            if not out_csv_name:
                out_csv_name = os.path.basename(csv)

            os.makedirs(out_csv_dir, exist_ok=True)
            image_list.to_csv(os.path.join(out_csv_dir, out_csv_name))
