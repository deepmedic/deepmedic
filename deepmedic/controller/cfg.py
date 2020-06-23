from pathlib import Path
from deepmedic.controller.assertion import assert_input_model_config
from deepmedic.config.input import InputModelConfig


class ModelConfigCfgController:
    @classmethod
    def run(cls, cfg_path: Path):
        input_config = InputModelConfig.from_cfg_file(cfg_path)
        assert_input_model_config(input_config)
        model_config = input_config.to_model_config()
        return model_config
