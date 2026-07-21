#!/usr/bin/env python3
"""Unit tests for section_preprocess (boilerplate gate before field detection)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from section_preprocess import (  # noqa: E402
    DropReason,
    filter_sections,
    should_keep_section,
)


ROOT = SCRIPT_DIR.parents[4]


class TestSectionPreprocess(unittest.TestCase):
    def test_consent_agreement_prose_dropped_as_disclaimer(self) -> None:
        text = (
            'The words "we," "us," "our" and "ours" as used below refer to us, the dealer, '
            "and to the financial institution(s) selected to receive your application. "
            "In accordance with the Fair Credit Reporting Act, you authorize that such "
            "financial institutions may submit your applications to other financial institutions. "
            "You agree that we may obtain a consumer credit report periodically from one or more "
            "consumer reporting agencies (credit bureaus). You consent to receive autodialed calls "
            "at the following number(s) (843)377-7187. The dealer and the financial institutions "
            "may monitor and record telephone calls regarding your account for quality assurance."
        )
        kept, reason = should_keep_section(text, document_type="credit_application")
        self.assertFalse(kept)
        self.assertEqual(reason, DropReason.DISCLAIMER)

    def test_6bf89f71_p13_all_sections_dropped(self) -> None:
        sections_json = (
            ROOT
            / "wa577_gallery/credit_app_sections/dealertrack/6bf89f71/"
            "6bf89f71_p13_sections/6bf89f71_p13_sections.json"
        )
        payload = json.loads(sections_json.read_text())
        kept, dropped = filter_sections(payload["sections"], document_type="credit_application")
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(dropped), 3)
        reasons = {s["index"]: s["preprocess"]["reason"] for s in dropped}
        self.assertEqual(reasons[0], DropReason.LEGAL_NOTICE.value)
        self.assertEqual(reasons[1], DropReason.LEGAL_NOTICE.value)
        self.assertEqual(reasons[2], DropReason.FOOTER.value)

    def test_6f99d76d_s5_dropped_s0_kept(self) -> None:
        sections_json = (
            ROOT
            / "wa577_gallery/credit_app_sections/dealertrack/6f99d76d/"
            "6f99d76d_p2_sections/6f99d76d_p2_sections.json"
        )
        payload = json.loads(sections_json.read_text())
        by_index = {s["index"]: s for s in payload["sections"]}

        kept_s0, _ = should_keep_section(
            by_index[0]["text"], document_type="credit_application"
        )
        kept_s5, reason_s5 = should_keep_section(
            by_index[5]["text"], document_type="credit_application"
        )
        self.assertTrue(kept_s0)
        self.assertFalse(kept_s5)
        self.assertEqual(reason_s5, DropReason.DISCLAIMER)

    def test_form_block_with_filled_values_kept(self) -> None:
        text = (
            "Last Name First Name Social Security Number\n"
            "SMITH JOHN 123-45-6789"
        )
        kept, reason = should_keep_section(text, document_type="credit_application")
        self.assertTrue(kept)
        self.assertIsNone(reason)

    def test_empty_form_structure_kept(self) -> None:
        text = (
            "Last Name First Name Middle Initial Social Security Number Birth Date\n"
            "________________ ________________ _____ _______________ __________"
        )
        kept, reason = should_keep_section(text, document_type="credit_application")
        self.assertTrue(kept)
        self.assertIsNone(reason)

    def test_filter_sections_annotates_preprocess(self) -> None:
        sections = [
            {"index": 0, "text": "FEDERAL NOTICES\nIMPORTANT INFORMATION ABOUT PROCEDURES"},
            {"index": 1, "text": "Last Name First Name\nDOE JANE"},
        ]
        kept, dropped = filter_sections(sections, document_type="credit_application")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["index"], 1)
        self.assertTrue(kept[0]["preprocess"]["kept"])
        self.assertEqual(len(dropped), 1)
        self.assertFalse(dropped[0]["preprocess"]["kept"])
        self.assertEqual(dropped[0]["preprocess"]["reason"], DropReason.LEGAL_NOTICE.value)


if __name__ == "__main__":
    unittest.main()
