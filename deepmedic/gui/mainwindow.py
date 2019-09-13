from functools import partial
from PySide2 import QtWidgets, QtCore, QtGui

from deepmedic.gui.ui_mainwindow import Ui_DeepMedic2
from deepmedic.gui.config_window import enable_on_combobox_value
from deepmedic.gui.model_config_window import ModelConfigWindow
from deepmedic.gui.test_config_window import TestConfigWindow
from deepmedic.gui.train_config_window import TrainConfigWindow
from deepmedic.gui.config_utils import p, file_open

import time

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_DeepMedic2()
        self.ui.setupUi(self)

        self.ui.model_config_search_button.clicked.connect(partial(file_open, self, self.ui.model_config_path,
                                                                   text='Choose model config file'))
        self.ui.train_config_search_button.clicked.connect(partial(file_open, self, self.ui.train_config_path,
                                                                   text='Choose train config file'))
        self.ui.test_config_search_button.clicked.connect(partial(file_open, self, self.ui.test_config_path,
                                                                  text='Choose test config file'))
        self.model_config_window = ModelConfigWindow(self)
        self.train_config_window = TrainConfigWindow(self)
        self.test_config_window = TestConfigWindow(self)

        self.ui.model_config_create_button.clicked.connect(self.open_model_config_window)
        self.ui.train_config_create_button.clicked.connect(self.open_train_config_window)
        self.ui.test_config_create_button.clicked.connect(self.open_test_config_window)

        self.ui.load_path_search_button.clicked.connect(partial(file_open, self, self.ui.load_path,
                                                                text='Choose model checkpoint prefix'))
        self.ui.run_button.clicked.connect(self.run_deepmedic)
        self.ui.stop_button.setEnabled(False)
        self.ui.stop_button.clicked.connect(self.stop_deepmedic)
        self.ui.device_combobox.currentTextChanged.connect(self.enable_dev_num)

        self.enable_train_test()
        self.ui.session_combobox.currentTextChanged.connect(self.enable_train_test)

        self.ui.model_config_path.setText(
            '/vol/biomedic2/bgmarque/deepmedic/examples/configFiles/tinyCnn/model/modelConfig.cfg')
        self.ui.train_config_path.setText(
            '/vol/biomedic2/bgmarque/deepmedic/examples/configFiles/tinyCnn/train/trainConfig.cfg')

        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_read)
        self.process.readyReadStandardError.connect(self.stderr_read)
        self.process.started.connect(lambda: self.deep_medic_start(True))
        self.process.finished.connect(lambda: self.deep_medic_start(False))

    def deep_medic_start(self, started):
        self.ui.run_button.setEnabled(not started)
        self.ui.stop_button.setEnabled(started)

    def open_model_config_window(self):
        self.model_config_window.show()

    def open_train_config_window(self):
        self.train_config_window.show()

    def open_test_config_window(self):
        self.test_config_window.show()

    def append(self, text):
        cursor = self.ui.output_log.textCursor()
        cursor.movePosition(cursor.End)
        # the scrollBar is only moved to the bottom if the user has not moved it from the bottom, this is so the user
        # can read through the log without the window constantly jumping
        if self.ui.output_log.verticalScrollBar().value() == self.ui.output_log.verticalScrollBar().maximum():
            change_scrollbar = True
        else:
            change_scrollbar = False
        cursor.insertText(text)
        if change_scrollbar:
            self.ui.output_log.verticalScrollBar().setValue(self.ui.output_log.verticalScrollBar().maximum())

    def stdout_read(self):
        text = str(self.process.readAllStandardOutput(), encoding='utf-8')
        self.append(text)

    def stderr_read(self):
        text = str(self.process.readAllStandardError(), encoding='utf-8')
        self.append(text)

    def enable_train_test(self):
        enable_on_combobox_value(self.ui.session_combobox, 'Train', [self.ui.train_config_label,
                                                                     self.ui.train_config_path,
                                                                     self.ui.train_config_search_button])
        enable_on_combobox_value(self.ui.session_combobox, 'Test', [self.ui.test_config_label,
                                                                    self.ui.test_config_path,
                                                                    self.ui.test_config_search_button])

    def enable_dev_num(self):
        type(self.ui.device_num_text)
        enable_on_combobox_value(self.ui.device_combobox, 'GPU', [self.ui.device_num_label, self.ui.device_num_text])

    def run_deepmedic(self):
        model_arg = self.ui.model_config_path.text()
        load_arg = self.ui.load_path.text()

        if str(self.ui.session_combobox.currentText()) == 'Test':
            sess_type = 'test'
            train_test_arg = self.ui.test_config_path.text()
        else:
            sess_type = 'train'
            train_test_arg = self.ui.train_config_path.text()

        if str(self.ui.device_combobox.currentText()) == 'GPU':
            dev_arg = 'cuda' + self.ui.device_num_text.text()
        else:
            dev_arg = 'cpu'

        # params = ['nohup', '/vol/biomedic2/bgmarque/deepmedic/deepMedicRun',
        #           '-model', model_arg, '-' + sess_type, train_test_arg, '-dev', dev_arg]

        params = ['/vol/biomedic2/bgmarque/deepmedic/deepMedicRun',
                  '-model', model_arg, '-' + sess_type, train_test_arg, '-dev', dev_arg]

        if load_arg:
            params += ['-load', load_arg]

        if self.ui.resetopt_checkbox.isChecked():
            params += ['-resetopt']

        # self.process = subprocess.Popen(params)
        self.process.start('python', params)

    def stop_deepmedic(self):
        self.process.terminate()
        self.ui.resetopt_checkbox.show()
        self.ui.resetopt_label.show()
