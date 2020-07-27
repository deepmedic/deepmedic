import unittest
import subprocess
from tests import TEST_DIR


class DeepMedicRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = TEST_DIR.parent.joinpath("examples", "configFiles", "tinyCnn", "model", "modelConfig.cfg")
        self.train_cfg_path = TEST_DIR.parent.joinpath("examples", "configFiles", "tinyCnn", "train", "trainConfigWithValidation.cfg")
        self.run_script_path = TEST_DIR.parent.joinpath("deepMedicRun")

    def test_example_train(self):
        subprocess.run(
            ["python", str(self.run_script_path), "-model", str(self.model_path), "-train", str(self.train_cfg_path)],
            shell=True, check=True
        )
