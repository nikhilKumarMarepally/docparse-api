import unittest
from unittest import mock

from app.extract.section_gate import gate_min_confidence, parse_gate_response, section_gate_enabled


class TestSectionGate(unittest.TestCase):
    def test_parse_extractable_confidence(self) -> None:
        r = parse_gate_response(
            {"extractable": True, "confidence": 0.92},
            provider="local",
            model="qwen2.5:3b",
        )
        self.assertTrue(r.extractable)
        self.assertAlmostEqual(r.confidence, 0.92)
        self.assertTrue(r.passes_threshold(0.5))

    def test_low_confidence_fails_threshold(self) -> None:
        r = parse_gate_response(
            {"extractable": True, "confidence": 0.3},
            provider="local",
        )
        self.assertFalse(r.passes_threshold(0.5))

    def test_gate_off_by_default(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(section_gate_enabled())

    def test_default_min_confidence(self) -> None:
        self.assertEqual(gate_min_confidence(), 0.5)


if __name__ == "__main__":
    unittest.main()
