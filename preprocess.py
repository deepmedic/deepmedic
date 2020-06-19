import sys
from PySide2 import QtWidgets

from deepmedic.gui.preproc_config_window import PreprocConfigWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = PreprocConfigWindow()
    widget.show()

    sys.exit(app.exec_())
