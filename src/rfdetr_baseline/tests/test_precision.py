import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import resolve_amp


class PrecisionPolicyTests(unittest.TestCase):
    def test_turing_uses_fp32_even_if_bf16_emulation_is_reported(self):
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_capability',return_value=(7,5)), \
             patch('torch.cuda.is_bf16_supported',return_value=True):
            self.assertFalse(resolve_amp(True))

    def test_ampere_retains_native_bf16(self):
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_capability',return_value=(8,6)), \
             patch('torch.cuda.is_bf16_supported',return_value=True):
            self.assertTrue(resolve_amp(True))

    def test_explicit_fp32_is_preserved(self):
        self.assertFalse(resolve_amp(False))


if __name__ == '__main__':
    unittest.main()
