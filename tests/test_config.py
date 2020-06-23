import unittest
from tests import TEST_DIR, TEST_OUTPUT_DIR
# from deepmedic.frontEnd.configParsing.modelParams import ModelParameters
# from deepmedic.frontEnd.configParsing.modelConfig import ModelConfig
from deepmedic.logging import loggers
from deepmedic.controller.cfg import ModelConfigCfgController
from deepmedic.config.input import InputModelConfig
from deepmedic.config.model import ModelConfig, PathWayConfig, SubsampledPathwayConfig, FCLayersConfig
from copy import deepcopy

class ModelParamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tiny_model_cfg_path = TEST_DIR.parent.joinpath(
            "examples", "configFiles", "tinyCnn", "model", "modelConfig.cfg"
        )
        self.deepmedic_model_cfg_path = TEST_DIR.parent.joinpath(
            "examples", "configFiles", "deepMedic", "model", "modelConfig.cfg"
        )

    # def test_model_param(self):
    #     model_config = ModelConfig(str(self.deepmedic_model_cfg_path))
    #     TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    #     logger = loggers.Logger(str(TEST_OUTPUT_DIR.joinpath("model_param_deep.txt")))
    #     model_param = ModelParameters(logger, model_config)
    #     model_param.print_params()
    #
    #     model_config = ModelConfig(str(self.tiny_model_cfg_path))
    #     TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    #     logger = loggers.Logger(str(TEST_OUTPUT_DIR.joinpath("model_param_tiny.txt")))
    #     model_param = ModelParameters(logger, model_config)
    #     model_param.print_params()

    def test_model_config(self):
        input_config = InputModelConfig(
            model_name="deepmedic",
            n_classes=5,
            n_input_channels=2,
            n_fm_norm=[30, 30, 40, 40, 40, 40, 50, 50],
            kern_dim_norm=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            pad_mode_norm=None,
            resid_conn_layers_norm=[4, 6, 8],
            lower_rank_layers_norm=[],
            use_subsampled=True,
            n_fm_subs=[[30, 30, 40, 40, 40, 40, 50, 50]],
            kern_dim_subs=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            pad_mode_subs=None,
            subs_factor=[[3, 3, 3], [5, 5, 5]],
            resid_conn_layers_subs=[4, 6, 8],
            lower_rank_layers_subs=None,
            n_fm_fc=[250, 250],
            kern_dim_fc=[[3, 3, 3], [1, 1, 1], [1, 1, 1]],
            pad_mode_fc=['mirror', None, None],
            resid_conn_layers_fc=[2],
            seg_dim_train=[37, 37, 37],
            seg_dim_val=[17, 17, 17],
            seg_dim_infer=[45, 45, 45],
            drop_norm=[],
            drop_subs=[],
            drop_fc=[0.0, 0.5, 0.5],
            conv_w_init=["fanIn", 2],
            activ_func="prelu",
            bn_roll_av_batches=60,

        )
        model_config = input_config.to_model_config()
        # assert model config print params
        TEST_OUTPUT_DIR.joinpath("model_config_deep.txt").unlink()
        logger = loggers.Logger(str(TEST_OUTPUT_DIR.joinpath("model_config_deep.txt")))
        model_config.print_params(logger)
        self.assertTrue(TEST_OUTPUT_DIR.joinpath("model_config_deep.txt").exists())
        normal_pathway_config = PathWayConfig(
            n_FMs_per_layer=[30, 30, 40, 40, 40, 40, 50, 50],
            kernel_dims_per_layer=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                   [3, 3, 3]],
            pad_mode_per_layer=['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID'],
            dropout=[],
            apply_bn=False,
            mp_params=[[], [], [], [], [], [], [], []],
            res_conn=[3, 5, 7],
            lower_rank=[],
        )
        self.assertEqual(normal_pathway_config, model_config.normal_pathway_config)
        sub_config = SubsampledPathwayConfig(
            n_FMs_per_layer=[30, 30, 40, 40, 40, 40, 50, 50],
            kernel_dims_per_layer=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            pad_mode_per_layer=['VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID', 'VALID'],
            dropout=[],
            apply_bn=False,
            mp_params=[[], [], [], [], [], [], [], []],
            res_conn=[3, 5, 7],
            lower_rank=[],
            subsample_factor=None
        )
        sub_config1 = deepcopy(sub_config)
        sub_config1.subsample_factor = [3, 3, 3]
        sub_config2 = deepcopy(sub_config)
        sub_config2.subsample_factor = [5, 5, 5]

        fc_config = FCLayersConfig(
            n_FMs_per_layer=[250, 250],
            kernel_dims_per_layer=[[3, 3, 3], [1, 1, 1], [1, 1, 1]],
            pad_mode_per_layer=['mirror', None, None],
            dropout=[0.0, 0.5, 0.5],
            apply_bn=True,
            mp_params=[[], [], []],
            res_conn=[1],
            lower_rank=[],
        )
        expected_model_config = ModelConfig(
            model_name="deepmedic",
            n_classes=5,
            n_input_channels=2,
            normal_pathway_config=normal_pathway_config,
            use_subsampled_path=True,
            subsampled_pathway_configs=[sub_config1, sub_config2],
            fc_layers_config=fc_config,
            activation_function="prelu",
            conv_w_init_type=["fanIn", 2],
            n_batches_for_bn_mov_avg=60,
            segment_dim_train=[37, 37, 37],
            segment_dim_val=[17, 17, 17],
            segment_dim_inference=[45, 45, 45]
        )
        self.assertEqual(expected_model_config, model_config)
