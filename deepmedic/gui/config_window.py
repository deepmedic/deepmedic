import os
from functools import partial
from PySide2 import QtWidgets, QtCore
from deepmedic.gui.ui_config_create import UiConfig


def clear_all_texts(all_texts):
    [text.setText('') for text in all_texts]


def clear_all_comboboxes(all_comboboxes):
    [combobox.setCurrentIndex(0) for combobox in all_comboboxes]


def clear_all_checkboxes(all_checkboxes):
    [checkbox.setChecked(False) for checkbox in all_checkboxes]


def enable_on_combobox_value(combobox, text, to_enable):
    set_value = bool(str(combobox.currentText()) == text)
    for element in to_enable:
        element.setEnabled(set_value)


def get_widget_type(elem_type):
    if elem_type == 'lineedit':
        return QtWidgets.QLineEdit
    if elem_type == 'checkbox':
        return QtWidgets.QCheckBox
    if elem_type == 'combobox':
        return QtWidgets.QComboBox


class ConfigWindow(QtWidgets.QMainWindow):
    def __init__(self, Config, window_type='', parent=None):
        super(ConfigWindow, self).__init__(parent)
        self.filename = None
        self.window_type = window_type
        self.Config = Config

        self.ui = UiConfig()
        self.ui.setup_ui(self, self.Config, window_type)

        self.ui.action_close.triggered.connect(self.close)
        self.ui.action_load.triggered.connect(self.load_config)
        self.ui.action_open.triggered.connect(self.open_config)
        self.ui.action_save_as.triggered.connect(self.save_as_config)
        self.ui.action_save.triggered.connect(partial(self.save_config))
        self.ui.action_clear_all.triggered.connect(self.clear_all)

        self.ui.save_button.clicked.connect(self.save_as_config)

        # self.clear_all()

        self.string_elems = self.get_all_string_elems()

        self.model_config_dict = self.create_model_config_dict()

    def get_all_string_elems(self):
        string_elems = []

        for section in self.Config.config_data.get_sorted_sections():
            for elem in section.get_sorted_elems():
                if elem.elem_type == 'String':
                    string_elems += [elem.name]

        return string_elems

    def show_messagebox(self, box_title=None, text=None, info=None, icon=None):
        msg = QtWidgets.QMessageBox(self)
        if icon:
            msg.setIcon(icon)
        if box_title:
            msg.setWindowTitle(box_title)
        if text:
            msg.setText(text)
        if info:
            msg.setInformativeText(info)
        msg.exec_()

    def open_config(self):
        self.filename = self.load_config()
        self.setWindowTitle('DeepMedic2 - ' + os.path.basename(self.filename))

    def load_config(self):
        filename = self.get_open_filename(text='Load ' + self.window_type + ' Configuration',
                                          formats='DeepMedic Config Files (*.cfg);; All Files (*)')
        model_cfg = self.Config(filename)
        for name, value in self.model_config_dict.items():
            cfg_value = model_cfg[name]
            if cfg_value:
                if hasattr(self.Config, 'CONV_W_INIT') and name == self.Config.CONV_W_INIT:
                    index = value[0].findText(cfg_value[0], QtCore.Qt.MatchFixedString)
                    if index < 0:
                        index = 0
                    value[0].setCurrentIndex(index)
                    if value[0].currentText():
                        value[1][value[0].currentText()].setText(str(cfg_value[1]))
                elif value.__class__ == QtWidgets.QLineEdit:
                    value.setText(str(cfg_value))
                elif value.__class__ == QtWidgets.QCheckBox:
                    value.setChecked(bool(cfg_value))
                elif value.__class__ == QtWidgets.QComboBox:
                    index = value.findText(cfg_value, QtCore.Qt.MatchFixedString)
                    if index < 0:
                        index = 0
                    value.setCurrentIndex(index)
        return filename

    def save_config(self, filename=None):
        if not filename:
            filename = self.filename

        if filename:
            with open(filename, 'w+') as f:
                f.write('# Created automatically using the DeepMedic2 GUI\n')
                for name, value in self.model_config_dict.items():
                    value_text = None
                    if hasattr(self.Config, 'CONV_W_INIT') and name == self.Config.CONV_W_INIT:
                        if value[0].currentText():
                            value_text = [value[0].currentText(), int(value[1][value[0].currentText()].text())]
                    elif value.__class__ == QtWidgets.QLineEdit:
                        value_text = value.text()
                    elif value.__class__ == QtWidgets.QCheckBox:
                        value_text = value.isChecked()
                    elif value.__class__ == QtWidgets.QComboBox:
                        value_text = value.currentText()
                    if name in self.string_elems:
                        value_text = '"' + value_text + '"'
                    if value_text:
                        f.write(str(name) + ' = ' + str(value_text) + '\n')
                    print(name)
                f.close()
        else:
            self.show_messagebox(box_title="Error saving config file",
                                 text="No file was open",
                                 info="Please open a file before saving or use "
                                      "Save As to choose a path to save your file in.",
                                 icon=QtWidgets.QMessageBox.Warning)

    def save_as_config(self):
        filename = self.get_save_filename(text='Save ' + self.window_type + ' Configuration File',
                                          formats='DeepMedic Config Files (*.cfg);; All Files (*)')

        self.save_config(filename)

    def clear_all(self):
        clear_all_texts(self.findChildren(QtWidgets.QLineEdit))
        clear_all_comboboxes(self.findChildren(QtWidgets.QComboBox))
        clear_all_checkboxes(self.findChildren(QtWidgets.QCheckBox))

    def get_open_filename(self, text='Search', path='.', formats='All Files (*)'):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, text, path, formats)
        return filename

    def get_save_filename(self, text='Search', path='.', formats='All Files (*)'):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, text, path, formats)
        return filename

    def create_model_config_dict(self):
        config_dict = {}
        for section in self.Config.config_data.get_sorted_sections():
            for elem in section.get_sorted_elems():
                if hasattr(self.Config, 'CONV_W_INIT') and elem.elem_type == 'Conv_w':
                    conv_w_dict = {}
                    for value, sub_elem in self.conv_w_init_elem.options.items():
                        qwidget = get_widget_type(sub_elem.widget_type)
                        name = self.conv_w_init_elem.section.name + '_' + sub_elem.name
                        conv_w_dict[value] = self.findChild(qwidget, name + '_' + sub_elem.widget_type)

                    config_dict[elem.name] = \
                        (self.findChild(QtWidgets.QComboBox, elem.section.name + '_' + elem.name + '_combobox'),
                         conv_w_dict)
                else:
                    qwidget = get_widget_type(elem.widget_type)
                    config_dict[elem.name] = \
                        self.findChild(qwidget, elem.section.name + '_' + elem.name + '_' + elem.widget_type)
        return config_dict
