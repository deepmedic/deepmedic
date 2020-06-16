from pathlib import Path
from collections import defaultdict

from deepmedic.controller import input_to_model_config
from deepmedic.controller.assertion import assert_input_model_config
from deepmedic.config.input import InputModelConfig


def input_model_config_from_cfg_file(cfg_path: Path) -> InputModelConfig:
    cfg = defaultdict(lambda: None)
    exec(open(str(cfg_path)).read(), cfg)
    input_config = InputModelConfig(
        model_name=cfg["modelName"],
        n_classes=cfg["numberOfOutputClasses"],
        n_input_channels=cfg["numberOfInputChannels"],
        n_fm_norm=cfg["numberFMsPerLayerNormal"],
        kern_dim_norm=cfg["kernelDimPerLayerNormal"],
        pad_mode_norm=cfg["padTypePerLayerNormal"],
        resid_conn_layers_norm=cfg["layersWithResidualConnNormal"],
        lower_rank_layers_norm=cfg["lowerRankLayersNormal"],
        use_subsampled=cfg["useSubsampledPathway"],
        n_fm_subs=cfg["numberFMsPerLayerSubsampled"],
        kern_dim_subs=cfg["kernelDimPerLayerSubsampled"],
        pad_mode_subs=cfg["padTypePerLayerSubsampled"],
        subs_factor=cfg["subsampleFactor"],
        resid_conn_layers_subs=cfg["layersWithResidualConnSubsampled"],
        lower_rank_layers_subs=cfg["lowerRankLayersSubsampled"],
        n_fm_fc=cfg["numberFMsPerLayerFC"],
        kern_dim_fc=cfg["kernelDimPerLayerFC"],
        pad_mode_fc=cfg["padTypePerLayerFC"],
        resid_conn_layers_fc=cfg["layersWithResidualConnFC"],
        seg_dim_train=cfg["segmentsDimTrain"],
        seg_dim_val=cfg["segmentsDimVal"],
        seg_dim_infer=cfg["segmentsDimInference"],
        drop_norm=cfg["dropoutRatesNormal"],
        drop_subs=cfg["dropoutRatesSubsampled"],
        drop_fc=cfg["dropoutRatesFc"],
        conv_w_init=cfg["convWeightsInit"],
        activ_func=cfg["activationFunction"],
        bn_roll_av_batches=cfg["rollAverageForBNOverThatManyBatches"],
    )
    return input_config


class ModelConfigCfgController:
    @classmethod
    def run(cls, cfg_path: Path):
        input_config = input_model_config_from_cfg_file(cfg_path)
        assert_input_model_config(input_config)
        model_config = input_to_model_config(input_config)
        return model_config
