import os
from functools import partial
from PySide2 import QtWidgets, QtCore
from deepmedic.gui.ui_config_create import UiConfig
from deepmedic.gui.config_utils import *


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
        self.ui.action_save.triggered.connect(self.save_config)
        self.ui.action_clear_all.triggered.connect(self.clear_all)

        self.ui.save_button.clicked.connect(self.save_as_config)

        # self.clear_all()

        self.model_config_dict = self.create_config_dict()

        self.string_elems = self.get_all_string_elems()
        self.connect_parent_statechanged()

        self.connect_all_search_buttons()

    def connect_all_search_buttons(self):
        all_buttons = self.findChildren(QtWidgets.QPushButton)
        for button in all_buttons:
            if not button == self.ui.save_button:
                elem = self.Config.config_data.get_elem('_'.join(button.objectName().split('_')[1:-1]))
                if elem is not None:
                    text_input = self.findChild(QtWidgets.QLineEdit, elem.section.name + '_' + elem.name + "_lineedit")
                    if text_input:
                        button.clicked.connect(partial(get_search_function(elem.elem_type),
                                                       self,
                                                       text_input,
                                                       'Search ' + elem.elem_type + ' (' + elem.description + ')'))

    def connect_parent_statechanged(self):
        elems_with_children = self.Config.config_data.get_elems_with_children()
        for parent, children in elems_with_children.items():
            if parent.elem_type == 'Bool':
                parent_widget = self.model_config_dict[parent.name]
                parent_widget.stateChanged.connect(partial(self.hide_children, parent_widget, children))
                self.hide_children(parent_widget, children)  # initialise hidden/visible

    def hide_children(self, parent_widget, children, set_value=None):
        if set_value is None:
            set_value = bool(parent_widget.isChecked())
        set_value = (set_value and bool(parent_widget.isEnabled()))
        for child in children:
            widget = self.findChild(QtWidgets.QLabel, child.section.name + '_' + child.name + "_label")
            if widget:
                widget.setEnabled(set_value)

            widget = self.model_config_dict[child.name]
            if widget:
                widget.setEnabled(set_value)
                if child.elem_type == 'Bool':
                    grandchildren = self.Config.config_data.get_children(child.name)
                    if grandchildren:
                        self.hide_children(widget, grandchildren)

            widget = self.findChild(QtWidgets.QPushButton, child.section.name + '_' + child.name + "_button")
            if widget:
                widget.setEnabled(set_value)

    def print_all_widget_names(self):
        children = self.findChildren(QtWidgets.QCheckBox)
        for child in children:
            if child.objectName().startswith('preproc'):
                print(child.objectName() + ' ' + str(child))

    def get_all_string_elems(self):
        string_elems = []

        for section in self.Config.config_data.get_sorted_sections():
            for elem in section.get_sorted_elems():
                if elem.elem_type in ['String', 'File', 'Folder', 'Files']:
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

    def get_dictionary_value(self, value):
        dict_value = {}
        for sub_name, sub_value in value.items():
            dict_value[sub_name] = self.get_text_value(sub_name, sub_value)
        return dict_value

    def set_dictionary_value(self, value, cfg_value):
        for sub_name, sub_value in value.items():
            self.set_text_value(sub_name, sub_value, cfg_value[sub_name])

    def get_text_value(self, name, value):
        value_text = None
        if hasattr(self.Config, 'CONV_W_INIT') and name == self.Config.CONV_W_INIT:
            print(value)
            if value[0].currentText():
                value_text = [value[0].currentText(), int(value[1][value[0].currentText()].text())]
        elif value.__class__ == QtWidgets.QLineEdit:
            value_text = value.text()
        elif value.__class__ == QtWidgets.QCheckBox:
            value_text = value.isChecked()
        elif value.__class__ == QtWidgets.QComboBox:
            value_text = value.currentText()
        elif value.__class__ == dict:
            value_text = self.get_dictionary_value(value)

        if value_text and name in self.string_elems:
            value_text = '"' + value_text + '"'
        else:
            value_text = num(value_text)

        return value_text

    def set_text_value(self, name, value, cfg_value):
        if cfg_value is not None or value.__class__ == QtWidgets.QCheckBox:
            if hasattr(self.Config, 'CONV_W_INIT') and name == self.Config.CONV_W_INIT:
                print(value)
                index = value[0].findText(cfg_value[0], QtCore.Qt.MatchFixedString)
                if index < 0:
                    index = 0
                value[0].setCurrentIndex(index)
                if value[0].currentText():
                    value[1][value[0].currentText()].setText(str(cfg_value[1]))
            elif value.__class__ == QtWidgets.QLineEdit:
                value.setText(str(cfg_value))
            elif value.__class__ == QtWidgets.QCheckBox:
                if cfg_value is None:
                    cfg_value = self.Config.config_data.get_elem(name).default
                value.setChecked(bool(cfg_value))
            elif value.__class__ == QtWidgets.QComboBox:
                index = value.findText(str(cfg_value), QtCore.Qt.MatchFixedString)
                if index < 0:
                    index = 0
                value.setCurrentIndex(index)
            elif value.__class__ == dict:
                self.set_dictionary_value(value, cfg_value)
                pass

    def load_config(self):
        filename = self.get_open_filename(text='Load ' + self.window_type + ' Configuration',
                                          formats='DeepMedic Config Files (*.cfg);; All Files (*)')
        model_cfg = self.Config(filename)
        for name, value in self.model_config_dict.items():
            cfg_value = model_cfg[name]
            self.set_text_value(name, value, cfg_value)

        return filename

    def save_config(self, filename=None):
        if not filename:
            filename = self.filename

        if filename:
            with open(filename, 'w+') as f:
                f.write('# Created automatically using the DeepMedic2 GUI\n')
                for name, value in self.model_config_dict.items():
                    value_text = self.get_text_value(name, value)
                    if value_text:
                        f.write(str(name) + ' = ' + str(value_text) + '\n')
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

    def create_config_entry(self, elem, prefix=''):
        if hasattr(self.Config, 'CONV_W_INIT') and elem.elem_type == 'Conv_w':
            conv_w_dict = {}
            for value, sub_elem in elem.options.items():
                qwidget = get_widget_type(sub_elem.widget_type)
                if prefix:
                    name = '_' + sub_elem.name
                else:
                    name = sub_elem.name
                conv_w_dict[value] = self.findChild(qwidget, prefix + name + '_' + sub_elem.widget_type)
            return self.findChild(QtWidgets.QComboBox, prefix + '_combobox'), conv_w_dict

        elif elem.widget_type == 'multiple':
            mult_dict = {}
            for name, sub_elem in elem.options.items():
                mult_dict[name] = self.create_config_entry(sub_elem,
                                                           prefix=prefix + '_' + sub_elem.name)
            return mult_dict
        else:
            qwidget = get_widget_type(elem.widget_type)
            return self.findChild(qwidget, prefix + '_' + elem.widget_type)

    def create_config_dict(self):
        config_dict = {}
        for section in self.Config.config_data.get_sorted_sections():
            for elem in section.get_sorted_elems():
                config_dict[elem.name] = self.create_config_entry(elem, prefix=elem.section.name + '_' + elem.name)
        return config_dict
