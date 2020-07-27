import unittest
from copy import deepcopy
from tests import TEST_OUTPUT_DIR
import tensorflow as tf

from deepmedic.neuralnet.cnn3d import Cnn3d
from deepmedic.neuralnet.pathways import NormalPathway, SubsampledPathway, FCPathway
from deepmedic.config.model import ModelConfig, PathWayConfig, SubsampledPathwayConfig, FCLayersConfig
from deepmedic.logging.loggers import Logger
from deepmedic.neuralnet.blocks import SoftmaxBlock


class Cnn3dTest(unittest.TestCase):
    def setUp(self) -> None:
        normal_pathway_config = PathWayConfig(
            n_FMs_per_layer=[30, 30, 30, 30, 30, 30, 30, 30],
            kernel_dims_per_layer=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],
                                   [3, 3, 3]],
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
        self.config = ModelConfig(
            model_name="deepmedic",
            n_classes=5,
            n_input_channels=2,
            normal_pathway_config=normal_pathway_config,
            subsampled_pathway_configs=[sub_config1, sub_config2],
            fc_layers_config=fc_config,
            activation_function="prelu",
            conv_w_init_type=["fanIn", 2],
            n_batches_for_bn_mov_avg=60,
            segment_dim_train=[37, 37, 37],
            segment_dim_val=[17, 17, 17],
            segment_dim_inference=[45, 45, 45]
        )
        if TEST_OUTPUT_DIR.joinpath("test_cnn3d.txt").exists():
            TEST_OUTPUT_DIR.joinpath("test_cnn3d.txt").unlink()
        self.logger = Logger(str(TEST_OUTPUT_DIR.joinpath("test_cnn3d.txt")))

    def test_init(self):
        cnn3d = Cnn3d(config=self.config, log=self.logger)
        self.assertEqual(len(cnn3d.pathways), 4)
        self.assertIsInstance(cnn3d.pathways[0], NormalPathway)
        self.assertIsInstance(cnn3d.pathways[1], SubsampledPathway)
        self.assertIsInstance(cnn3d.pathways[2], SubsampledPathway)
        self.assertIsInstance(cnn3d.pathways[3], FCPathway)
        self.assertIsInstance(cnn3d.final_target_layer, SoftmaxBlock)

    def test_build(self):

        with tf.Graph().as_default():
            with tf.device("cpu"):
                cnn3d = Cnn3d(config=self.config, log=self.logger)
                with tf.compat.v1.variable_scope("net"):
                    cnn3d.build()
                    # I have now created the CNN graph. But not yet the Optimizer's graph.
                    inp_plchldrs_train, inp_shapes_per_path_train = cnn3d.create_input_placeholders("train")
                    inp_plchldrs_val, inp_shapes_per_path_val = cnn3d.create_input_placeholders("val")
                    inp_plchldrs_test, inp_shapes_per_path_test = cnn3d.create_input_placeholders("test")
                    p_y_given_x_train = cnn3d.apply(inp_plchldrs_train, "train", "train", verbose=True)
                    p_y_given_x_val = cnn3d.apply(inp_plchldrs_val, "infer", "val", verbose=True)
                    p_y_given_x_test = cnn3d.apply(inp_plchldrs_test, "infer", "test", verbose=True)
        self.assertIsInstance(p_y_given_x_train, tf.Tensor)
        self.assertListEqual(p_y_given_x_train.shape.as_list(), [None, 5, 21, 21, 21])
        self.assertIsInstance(p_y_given_x_val, tf.Tensor)
        self.assertListEqual(p_y_given_x_val.shape.as_list(), [None, 5, 1, 1, 1])
        self.assertIsInstance(p_y_given_x_test, tf.Tensor)
        self.assertListEqual(p_y_given_x_test.shape.as_list(), [None, 5, 29, 29, 29])
        params = cnn3d.get_trainable_params(log=self.logger, indices_of_layers_per_pathway_type_to_freeze=[[], [], []])
        self.assertEqual(len(params), 82)
        self.assertIsInstance(params[0], tf.Variable)
        params = cnn3d.params_for_L1_L2_reg()
        self.assertEqual(len(params), 27)
        self.assertIsInstance(params[0], tf.Variable)

        self.assertListEqual(inp_shapes_per_path_train[0], [37, 37, 37])
        self.assertListEqual(cnn3d.calc_outp_dims_given_inp(inp_shapes_per_path_train[0]), [21, 21, 21])
        self.assertListEqual(cnn3d.calc_unpredicted_margin(inp_shapes_per_path_train[0]), [[8, 8] for _ in range(3)])

        self.assertListEqual(inp_shapes_per_path_train[1], [23, 23, 23])
        self.assertListEqual(cnn3d.calc_outp_dims_given_inp(inp_shapes_per_path_train[1]), [7, 7, 7])
        self.assertListEqual(cnn3d.calc_unpredicted_margin(inp_shapes_per_path_train[1]), [[8, 8] for _ in range(3)])

        self.assertListEqual(inp_shapes_per_path_train[2], [21, 21, 21])
        self.assertListEqual(cnn3d.calc_outp_dims_given_inp(inp_shapes_per_path_train[2]), [5, 5, 5])
        self.assertListEqual(cnn3d.calc_unpredicted_margin(inp_shapes_per_path_train[2]), [[8, 8] for _ in range(3)])

        self.assertListEqual(inp_shapes_per_path_train[2], [21, 21, 21])
        self.assertListEqual(cnn3d.calc_outp_dims_given_inp(inp_shapes_per_path_train[3]), [5, 5, 5])
        self.assertListEqual(cnn3d.calc_unpredicted_margin(inp_shapes_per_path_train[3]), [[8, 8] for _ in range(3)])

    def test_get_apis(self):
        cnn3d = Cnn3d(config=self.config, log=self.logger)
        self.assertEqual(cnn3d.get_num_subs_pathways(), 2)
        self.assertEqual(cnn3d.get_num_pathways_that_require_input(), 3)
