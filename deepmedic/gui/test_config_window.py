from PySide2 import QtWidgets

from deepmedic.gui.config_window import ConfigWindow
from deepmedic.frontEnd.configParsing.testConfig import TestConfig


class TestConfigWindow(ConfigWindow):
    def __init__(self, parent=None):
        super(TestConfigWindow, self).__init__(TestConfig, 'Test', parent)
