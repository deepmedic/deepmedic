from deepmedic.gui.config_window import ConfigWindow
from deepmedic.gui.ui_preproc_config_create import UiPreprocConfig
from deepmedic.frontEnd.configParsing.preprocConfig import PreprocConfig
from deepmedic.dataManagement.data_checks import run_checks

from PySide2 import QtWidgets, QtGui


class ProgressBar(object):
    def __init__(self, bar):
        self.bar = bar
        self.bar.setMinimum(0)
        self.bar.setValue(0)

    def increase_value(self):
        self.bar.setValue(self.bar.value() + 1)
        QtGui.qApp.processEvents()


class PreprocConfigWindow(ConfigWindow):
    def __init__(self, parent=None):
        super(PreprocConfigWindow, self).__init__(PreprocConfig, 'Preprocess Data', parent,
                                                  UiConfigClass=UiPreprocConfig)

        self.ui.data_checks_button.clicked.connect(self.run_data_checks)
        self.bar = ProgressBar(self.ui.data_checks_progress)
        self.bar.bar.hide()

    def run_data_checks(self):
        csv = self.findChild(QtWidgets.QLineEdit, 'data_inputCsv_lineedit').text()
        self.ui.data_checks_progress.show()
        check_text = run_checks(csv, csv=True, pixs=True, dims=True, dtypes=True, disable_tqdm=False, html=True,
                                progress=self.bar)
        self.ui.data_checks_text.setText(check_text)
