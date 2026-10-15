"""
完善后的单元测试套件 (test_quant_framework.py)
全面覆盖：
  - QuantConfig 配置校验、1-16bit 枚举、dict/file 序列化读取
  - BaseGGUFConverter 转换与已有 GGUF 输入识别
  - ImatrixGenerator 4 种模式及 UD 命名
  - GGUFQuantizer 1-16bit 命令构造与 UD 优化档位生成
  - QuantPipeline 端到端一键执行流程
"""

import unittest
import tempfile
import os
import shutil
import json
from unittest.mock import patch

from quant_framework.quant_config import QuantConfig, ImatrixMode, QuantType
from quant_framework.imatrix_generator import ImatrixGenerator
from quant_framework.gguf_quantizer import GGUFQuantizer, BaseGGUFConverter
from quant_framework.quant_pipeline import QuantPipeline


class TestQuantConfig(unittest.TestCase):

    def test_valid_config(self):
        cfg = QuantConfig(
            model_input_path="/tmp/test_model",
            output_dir="/tmp/test_out",
            imatrix_mode=ImatrixMode.NONE,
        )
        cfg.validate()

    def test_missing_input_dir(self):
        cfg = QuantConfig(model_input_path="", output_dir="/tmp/out")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_dataset_mode_missing_path(self):
        cfg = QuantConfig(
            model_input_path="/tmp/in",
            output_dir="/tmp/out",
            imatrix_mode=ImatrixMode.DATASET,
            calibration_dataset_path=None,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_existing_mode_missing_path(self):
        cfg = QuantConfig(
            model_input_path="/tmp/in",
            output_dir="/tmp/out",
            imatrix_mode=ImatrixMode.EXISTING,
            existing_imatrix_path=None,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_full_range_quant_types(self):
        """覆盖 1-bit 到 16-bit 各典型档位"""
        self.assertEqual(QuantType.IQ1_S.value, "IQ1_S")
        self.assertEqual(QuantType.IQ2_XXS.value, "IQ2_XXS")
        self.assertEqual(QuantType.Q3_K_M.value, "Q3_K_M")
        self.assertEqual(QuantType.Q4_K_M.value, "Q4_K_M")
        self.assertEqual(QuantType.Q5_K_M.value, "Q5_K_M")
        self.assertEqual(QuantType.Q6_K.value, "Q6_K")
        self.assertEqual(QuantType.Q8_0.value, "Q8_0")
        self.assertEqual(QuantType.BF16.value, "BF16")

    def test_from_dict_and_to_dict(self):
        d = {
            "model_input_path": "/tmp/model",
            "output_dir": "/tmp/out",
            "target_quants": ["Q4_K_M", "IQ2_XXS"],
            "imatrix_mode": "dataset",
            "calibration_dataset_path": "/tmp/calib.txt",
        }
        cfg = QuantConfig.from_dict(d)
        self.assertEqual(cfg.imatrix_mode, ImatrixMode.DATASET)
        self.assertEqual(len(cfg.target_quants), 2)
        out_dict = cfg.to_dict()
        self.assertEqual(out_dict["imatrix_mode"], "dataset")
        self.assertIn("Q4_K_M", out_dict["target_quants"])

    def test_from_json_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            cfg_file = os.path.join(temp_dir, "config.json")
            with open(cfg_file, "w") as f:
                json.dump({
                    "model_input_path": "/tmp/model.gguf",
                    "output_dir": "/tmp/out",
                    "imatrix_mode": "none"
                }, f)
            cfg = QuantConfig.from_file(cfg_file)
            self.assertEqual(cfg.model_input_path, "/tmp/model.gguf")
            self.assertEqual(cfg.imatrix_mode, ImatrixMode.NONE)
        finally:
            shutil.rmtree(temp_dir)


class _BaseFixture(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input_model")
        self.output_dir = os.path.join(self.test_dir, "output_quants")
        self.calib_file = os.path.join(self.test_dir, "calib.txt")

        os.makedirs(self.input_dir, exist_ok=True)
        with open(os.path.join(self.input_dir, "model-00001-of-00001.safetensors"), "wb") as f:
            f.write(b"MOCK_SAFETENSORS_HEADER")
        with open(self.calib_file, "w") as f:
            f.write("Sample calibration text for testing.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _cfg(self, **kwargs) -> QuantConfig:
        defaults = dict(
            model_input_path=self.input_dir,
            output_dir=self.output_dir,
            imatrix_mode=ImatrixMode.NONE,
            model_name="Qwen3.8-27B",
        )
        defaults.update(kwargs)
        return QuantConfig(**defaults)


class TestBaseGGUFConverter(_BaseFixture):

    def test_directory_init(self):
        conv = BaseGGUFConverter(self._cfg(), self.output_dir)
        self.assertTrue(os.path.exists(conv.dir_base))
        self.assertTrue(os.path.exists(conv.dir_logs))

    def test_check_prerequisites_safetensors(self):
        conv = BaseGGUFConverter(self._cfg(), self.output_dir)
        self.assertTrue(conv.check_prerequisites())

    def test_check_prerequisites_direct_gguf(self):
        gguf_file = os.path.join(self.test_dir, "Qwen-BF16.gguf")
        with open(gguf_file, "w") as f:
            f.write("GGUF")
        cfg = self._cfg(model_input_path=gguf_file)
        conv = BaseGGUFConverter(cfg, self.output_dir)
        self.assertTrue(conv.check_prerequisites())
        self.assertEqual(conv.convert(), gguf_file)

    @patch("quant_framework.gguf_quantizer.run_command")
    def test_convert_safetensors(self, mock_run):
        conv = BaseGGUFConverter(self._cfg(), self.output_dir)
        out = conv.convert(outtype="bf16")
        self.assertEqual(out, conv.base_gguf_path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("--outtype", cmd)
        self.assertIn("bf16", cmd)


class TestImatrixGenerator(_BaseFixture):

    def _gen(self, cfg, base_gguf="/fake/base.gguf"):
        return ImatrixGenerator(cfg, base_gguf_path=base_gguf, work_dir=self.output_dir)

    def test_none_mode_returns_none(self):
        gen = self._gen(self._cfg(imatrix_mode=ImatrixMode.NONE))
        self.assertIsNone(gen.generate())

    def test_existing_mode(self):
        imx = os.path.join(self.test_dir, "imatrix.gguf")
        with open(imx, "w") as f:
            f.write("IMATRIX")
        cfg = self._cfg(imatrix_mode=ImatrixMode.EXISTING, existing_imatrix_path=imx)
        gen = self._gen(cfg)
        self.assertEqual(gen.generate(), imx)

    @patch("quant_framework.imatrix_generator.run_command")
    def test_dataset_mode_cmd(self, mock_run):
        cfg = self._cfg(
            imatrix_mode=ImatrixMode.DATASET,
            calibration_dataset_path=self.calib_file,
            imatrix_gpu_layers=99,
            imatrix_chunks=50,
        )
        gen = self._gen(cfg)

        def fake_run(cmd, log_path=None):
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "w") as f:
                f.write("IMATRIX")
        mock_run.side_effect = fake_run

        result = gen.generate()
        self.assertTrue(result.endswith("imatrix-Qwen3.8-27B.gguf"))
        cmd = mock_run.call_args[0][0]
        self.assertIn("-ngl", cmd)
        self.assertIn("35", cmd)
        self.assertIn("-b", cmd)

    @patch("quant_framework.imatrix_generator.run_command")
    def test_self_gen_mode_naming(self, mock_run):
        cfg = self._cfg(imatrix_mode=ImatrixMode.SELF_GENERATED)
        gen = self._gen(cfg)

        def fake_run(cmd, log_path=None):
            out_idx = cmd.index("-o") + 1
            with open(cmd[out_idx], "w") as f:
                f.write("IMATRIX")
        mock_run.side_effect = fake_run

        result = gen.generate()
        self.assertTrue(result.endswith("imatrix-UD-Qwen3.8-27B.gguf"))


class TestGGUFQuantizer(_BaseFixture):

    def _make_base(self, work_dir):
        base = os.path.join(work_dir, "model-BF16.gguf")
        os.makedirs(os.path.dirname(base), exist_ok=True)
        with open(base, "w") as f:
            f.write("DUMMY_BF16")
        return base

    def _qz(self, cfg):
        return GGUFQuantizer(cfg, self.output_dir)

    def _fake_quantize(self, cmd, log_path=None):
        out_file = cmd[-2]
        with open(out_file, "w") as f:
            f.write("DUMMY_QUANT")

    @patch("quant_framework.gguf_quantizer.run_command")
    def test_quantize_with_imatrix_and_overrides(self, mock_run):
        mock_run.side_effect = self._fake_quantize
        base = self._make_base(self.test_dir)

        imatrix = os.path.join(self.test_dir, "imatrix.gguf")
        with open(imatrix, "w") as f:
            f.write("IMATRIX")

        cfg = self._cfg(
            target_quants=[QuantType.Q4_K_M],
            enable_layered_overrides=True,
            tensor_overrides={"blk.0.*": "Q6_K", "blk.64.nextn.*": "Q8_0"},
        )
        qz = self._qz(cfg)
        results = qz.quantize_all(base, imatrix_path=imatrix)

        self.assertIn("Q4_K_M_imatrix", results)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--imatrix", cmd)
        self.assertIn("--override-tensor", cmd)
        self.assertIn("blk.0.*=Q6_K", cmd)

    @patch("quant_framework.gguf_quantizer.run_command")
    def test_ud_quantize_naming(self, mock_run):
        mock_run.side_effect = self._fake_quantize
        base = self._make_base(self.test_dir)

        cfg = self._cfg(target_quants=[QuantType.Q4_K_M])
        qz = self._qz(cfg)
        results = qz.quantize_all(base, imatrix_path=None, is_ud_mode=True)
        self.assertIn("UD_Q4_K_M", results)
        self.assertTrue(results["UD_Q4_K_M"].endswith("Qwen3.8-27B-UD-Q4_K_M.gguf"))


class TestQuantPipeline(_BaseFixture):

    def test_pipeline_directory_initialization(self):
        pipeline = QuantPipeline(self._cfg())
        self.assertTrue(os.path.exists(pipeline.dir_base))
        self.assertTrue(os.path.exists(pipeline.dir_calib))
        self.assertTrue(os.path.exists(pipeline.dir_quant))
        self.assertTrue(os.path.exists(pipeline.dir_logs))

    @patch("quant_framework.gguf_quantizer.run_command")
    def test_pipeline_with_direct_gguf_input(self, mock_run):
        def fake_quantize(cmd, log_path=None):
            with open(cmd[-2], "w") as f:
                f.write("QUANT")
        mock_run.side_effect = fake_quantize

        gguf_file = os.path.join(self.test_dir, "direct-BF16.gguf")
        with open(gguf_file, "w") as f:
            f.write("GGUF_WEIGHT")

        cfg = self._cfg(
            model_input_path=gguf_file,
            target_quants=[QuantType.Q4_K_M],
            imatrix_mode=ImatrixMode.NONE,
            enable_layered_overrides=False,
        )
        pipeline = QuantPipeline(cfg)
        results = pipeline.run_all()
        self.assertIn("Q4_K_M", results)


if __name__ == "__main__":
    unittest.main()
