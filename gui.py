import sys
from PySide2 import QtWidgets

from deepmedic.gui.mainwindow import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MainWindow()
    widget.show()

    sys.exit(app.exec_())
