from PySide2 import QtWidgets

from deepmedic.gui.config_window import ConfigWindow, get_widget_type, enable_on_combobox_value
from deepmedic.frontEnd.configParsing.modelConfig import ModelConfig


class ModelConfigWindow(ConfigWindow):
    def __init__(self, parent=None):
        conv_w_init_name = ModelConfig.CONV_W_INIT
        self.conv_w_init_elem = ModelConfig.config_data.get_elem(conv_w_init_name)

        super(ModelConfigWindow, self).__init__(ModelConfig, 'Model', parent)

        self.conv_w_init_prefix = self.conv_w_init_elem.section.name + '_' + conv_w_init_name
        self.enable_kernel_field()  # init kernel option
        self.findChild(QtWidgets.QComboBox, self.conv_w_init_prefix + '_combobox').currentTextChanged.\
            connect(self.enable_kernel_field)

        sub_checkbox_name = ModelConfig.USE_SUBSAMPLED
        self.sub_prefix = ModelConfig.config_data.get_elem_section(sub_checkbox_name).name + '_' + sub_checkbox_name
        self.sub_checkbox = self.findChild(QtWidgets.QCheckBox, self.sub_prefix + '_checkbox')

        self.sub_checkbox.setChecked(True)
        self.sub_checkbox.stateChanged.connect(self.hide_sub)

    def enable_kernel_field(self):
        for value, elem in self.conv_w_init_elem.options.items():
            qwidget = get_widget_type(elem.widget_type)
            name = self.conv_w_init_elem.section.name + '_' + elem.name
            enable_on_combobox_value(self.findChild(QtWidgets.QComboBox, self.conv_w_init_prefix + '_combobox'),
                                     value,
                                     [self.findChild(qwidget, name + '_' + elem.widget_type),
                                      self.findChild(QtWidgets.QLabel, name + '_label')
                                      ]
                                     )

    def hide_sub(self):
        set_value = bool(self.sub_checkbox.isChecked())
        [element.setEnabled(set_value)
         for element in self.findChildren(QtWidgets.QLineEdit) + self.findChildren(QtWidgets.QLabel)
         if element.objectName().startswith('sub_')
         and not (element.objectName() == 'sub_label' or element.objectName() == 'sub_useSubsampledPathway_label')]
