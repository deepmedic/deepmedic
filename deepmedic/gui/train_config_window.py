from deepmedic.gui.config_window import ConfigWindow
from deepmedic.frontEnd.configParsing.trainConfig import TrainConfig


class TrainConfigWindow(ConfigWindow):
    def __init__(self, parent=None):
        super(TrainConfigWindow, self).__init__(TrainConfig, 'Train', parent)
