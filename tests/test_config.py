import unittest
from tests import TEST_DIR, TEST_OUTPUT_DIR
from deepmedic.config.input.exceptions import (
    FCKernDimFMLengthNotEqualException,
    NumClassInputException,
    NumChannelsInputException,
    NumFMNormInputException,
    KernelDimNormInputException,
    ModelCfgListOfListException,
    NumFMSubsampledInputException,
    KernelDimSubsampledInputException,
    NormAndSubsampledReceptiveFieldNotEqualException,
    SubsampleFactorInputException,
    SegmentsDimInputException,
    ConvWInitTypeInputException,
    ActivationFunctionInputException,
    ResConnectionInputException,
)
from deepmedic.logging import loggers
from deepmedic.controller.cfg import assert_input_model_config
from deepmedic.config.input import InputModelConfig
from deepmedic.config.model import ModelConfig, PathWayConfig, SubsampledPathwayConfig, FCLayersConfig
from copy import deepcopy


class InputModelConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tiny_model_cfg_path = TEST_DIR.parent.joinpath(
            "examples", "configFiles", "tinyCnn", "model", "modelConfig.cfg"
        )
        self.deepmedic_model_cfg_path = TEST_DIR.parent.joinpath(
            "examples", "configFiles", "deepMedic", "model", "modelConfig.cfg"
        )
        self.input_config = InputModelConfig(
            model_name="deepmedic",
            n_classes=5,
            n_input_channels=2,
            n_fm_norm=[30, 30, 30, 30, 30, 30, 30, 30],
            kern_dim_norm=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            pad_mode_norm=['norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm'],
            resid_conn_layers_norm=[4, 6, 8],
            lower_rank_layers_norm=[],
            use_subsampled=True,
            n_fm_subs=[[40, 40, 40, 40, 40, 40, 40, 40], [40, 40, 40, 40, 40, 40, 40, 40]],
            kern_dim_subs=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            pad_mode_subs=['subs', 'subs', 'subs', 'subs', 'subs', 'subs', 'subs', 'subs'],
            subs_factor=[[3, 3, 3], [5, 5, 5]],
            resid_conn_layers_subs=[3, 5, 7],
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

        normal_pathway_config = PathWayConfig(
            n_FMs_per_layer=[30, 30, 30, 30, 30, 30, 30, 30],
            kernel_dims_per_layer=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                   [3, 3, 3]],
            pad_mode_per_layer=['norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm', 'norm'],
            dropout=[],
            apply_bn=False,
            mp_params=[[], [], [], [], [], [], [], []],
            res_conn=[3, 5, 7],
            lower_rank=[],
        )
        sub_config = SubsampledPathwayConfig(
            n_FMs_per_layer=[40, 40, 40, 40, 40, 40, 40, 40],
            kernel_dims_per_layer=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                   [3, 3, 3]],
            pad_mode_per_layer=['subs', 'subs', 'subs', 'subs', 'subs', 'subs', 'subs', 'subs'],
            dropout=[],
            apply_bn=False,
            mp_params=[[], [], [], [], [], [], [], []],
            res_conn=[2, 4, 6],
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
        self.expected_model_config = ModelConfig(
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

    def test_to_model_config(self):

        model_config = self.input_config.to_model_config()
        # assert model config print params
        TEST_OUTPUT_DIR.joinpath("model_config_deep.txt").unlink()
        logger = loggers.Logger(str(TEST_OUTPUT_DIR.joinpath("model_config_deep.txt")))
        model_config.print_params(logger)
        self.assertTrue(TEST_OUTPUT_DIR.joinpath("model_config_deep.txt").exists())
        self.assertEqual(self.expected_model_config.normal_pathway_config, model_config.normal_pathway_config)
        self.assertEqual(self.expected_model_config, model_config)

        # if kern_dim_subs is None
        input_config = deepcopy(self.input_config)
        input_config.kern_dim_subs = None
        expected_model_config = deepcopy(self.expected_model_config)
        for subs_config in expected_model_config.subsampled_pathway_configs:
            subs_config.kernel_dims_per_layer = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                                 [3, 3, 3], [3, 3, 3]]
        self.assertEqual(expected_model_config, input_config.to_model_config())

        input_config = deepcopy(self.input_config)
        input_config.subs_factor = None
        expected_model_config = deepcopy(self.expected_model_config)
        subs_config = deepcopy(self.expected_model_config.subsampled_pathway_configs[0])
        subs_config.subsample_factor = [3, 3, 3]
        expected_model_config.subsampled_pathway_configs = [subs_config]
        self.assertEqual(expected_model_config, input_config.to_model_config())

        input_config = deepcopy(self.input_config)
        input_config.n_fm_subs = [[40, 40, 40, 40, 40, 40, 40, 40]]
        self.assertEqual(self.expected_model_config, input_config.to_model_config())

    def test_assert_input_config(self):

        input_config = deepcopy(self.input_config)
        input_config.n_classes = None
        self.assertRaises(NumClassInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_input_channels = None
        self.assertRaises(NumChannelsInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_input_channels = 0
        self.assertRaises(NumChannelsInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_norm = None
        self.assertRaises(NumFMNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_norm = []
        self.assertRaises(NumFMNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_norm = None
        self.assertRaises(KernelDimNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_norm = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                      [3, 3, 3, 3], [3, 3, 3]]
        self.assertRaises(KernelDimNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_norm = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                      [3, 3, 3]]
        self.assertRaises(KernelDimNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_subs = 40
        self.assertRaises(ModelCfgListOfListException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_subs = [[40, 40, 40, 40, 40, 40, 40, 40], 40]
        self.assertRaises(ModelCfgListOfListException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_subs = [[40, 40, 40, 40, 40, 40, 40, 40], [40]]
        self.assertRaises(NumFMSubsampledInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.n_fm_subs = [[40, 40, 40, 40, 40, 40, 40], [40, 40, 40, 40, 40, 40, 40]]
        input_config.kern_dim_subs = None
        self.assertRaises(KernelDimSubsampledInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_subs = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                      [3, 3, 3, 3], [3, 3, 3]]
        self.assertRaises(KernelDimSubsampledInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_norm = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                      [3, 3, 3]]
        self.assertRaises(KernelDimNormInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_subs = [[4, 4, 4], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                      [3, 3, 3], [3, 3, 3]]
        self.assertRaises(NormAndSubsampledReceptiveFieldNotEqualException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.subs_factor = [[3, 3, 3], [5]]
        self.assertRaises(SubsampleFactorInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.subs_factor = [[3, 3, 3], 5]
        self.assertRaises(ModelCfgListOfListException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.kern_dim_fc = [[3, 3, 3], [1, 1, 1]]
        self.assertRaises(FCKernDimFMLengthNotEqualException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.seg_dim_train = None
        self.assertRaises(SegmentsDimInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.seg_dim_train = [1, 1, 1]
        self.assertRaises(SegmentsDimInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.seg_dim_train = [1, 1, 1, 1]
        self.assertRaises(SegmentsDimInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.conv_w_init = ["test", None]
        self.assertRaises(ConvWInitTypeInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.activ_func = "oru"
        self.assertRaises(ActivationFunctionInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.resid_conn_layers_norm = [1, 2, 3]
        self.assertRaises(ResConnectionInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.resid_conn_layers_subs = [1, 2, 3]
        self.assertRaises(ResConnectionInputException, assert_input_model_config, input_config)

        input_config = deepcopy(self.input_config)
        input_config.resid_conn_layers_fc = [1, 2, 3]
        self.assertRaises(ResConnectionInputException, assert_input_model_config, input_config)
