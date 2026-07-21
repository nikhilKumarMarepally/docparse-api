#!/usr/bin/env python3
"""Unit tests for generic section content classifier."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from section_content_classifier import classify_section  # noqa: E402
from section_content_heuristics import (  # noqa: E402
    classify_content_types_heuristic,
    normalize_ocr_text,
)
from section_content_taxonomy import (  # noqa: E402
    field_path_to_content_types,
    route_content_types_to_sections,
)


class TestHeuristics(unittest.TestCase):
    def test_ssn_row_personal_identity(self) -> None:
        text = "Last Name First Name Social Security Number\nSMITH JOHN 123-45-6789"
        present, _ = classify_content_types_heuristic(text)
        self.assertIn("personal_identity", present)

    def test_vin_block_vehicle_description(self) -> None:
        text = "Year Make Vehicle Identification Number\n2020 TOYOTA 1HGBH41JXMN109186"
        present, _ = classify_content_types_heuristic(text)
        self.assertIn("vehicle_description", present)

    def test_footer_not_signature(self) -> None:
        text = "Credit Application Creation Time: 01/15/2026 Printed: 01/15/2026 Page 1 of 3"
        present, _ = classify_content_types_heuristic(text)
        self.assertNotIn("signature_authorization", present)

    def test_signature_row(self) -> None:
        text = "Credit Application Signature\nApplicant: By __________ Date __________"
        present, _ = classify_content_types_heuristic(text)
        self.assertIn("signature_authorization", present)

    def test_trade_in_not_primary_vehicle(self) -> None:
        text = "Trade-In Year Make Vehicle Identification Number\n2015 HONDA 2HGFG3B54CH501234"
        present, _ = classify_content_types_heuristic(text)
        self.assertIn("trade_in_vehicle", present)

    def test_homoglyph_normalize(self) -> None:
        # Cyrillic 'с' and 'т' often OCR as CT state code corruption
        t = normalize_ocr_text("Stamford сt 06907")
        self.assertIn("stamford", t)

    def test_business_block(self) -> None:
        text = (
            "Trade Name of Business Tax ID\n"
            "Springdale Developers LLC 33-3229251"
        )
        present, _ = classify_content_types_heuristic(text)
        self.assertIn("business_entity", present)


class TestTaxonomy(unittest.TestCase):
    def test_field_path_mapping(self) -> None:
        self.assertIn("personal_identity", field_path_to_content_types("applicants.applicant1.ssn"))
        self.assertIn("vehicle_description", field_path_to_content_types("vin"))
        self.assertIn("trade_in_vehicle", field_path_to_content_types("trade_in_vin"))

    def test_route_credit_app(self) -> None:
        sections = route_content_types_to_sections(
            ["personal_identity", "employment_income"],
            "credit_application",
        )
        self.assertTrue(any("applicant" in s for s in sections))


class TestClassifier(unittest.TestCase):
    def test_classify_with_document_type(self) -> None:
        result = classify_section(
            "Last Name First Name DOB SSN\nDOE JANE 01/01/1990 111-22-3333",
            document_type="credit_application",
        )
        self.assertIn("personal_identity", result.content_types)
        self.assertEqual(result.method, "heuristic")

    def test_field_output_personal_block(self) -> None:
        result = classify_section(
            "Last Name First Name Present Address City State Zip\n"
            "DOE JANE 123 Main St Stamford CT 06907",
            document_type="credit_application",
        )
        self.assertIn("fields", result.to_dict())
        fields = result.fields
        self.assertIn("applicants.applicant1.last_name", fields)
        self.assertTrue(
            "applicants.applicant1.address" in fields
            or "applicants.applicant1.address.street_address" in fields
        )

    def test_field_output_ssn_shape(self) -> None:
        result = classify_section(
            "Social Security Number\n123-45-6789",
            document_type="credit_application",
        )
        self.assertIn("applicants.applicant1.ssn", result.fields)

    def test_field_output_co_applicant_scope(self) -> None:
        result = classify_section(
            "Co-Applicant Last Name First Name\nSMITH JOHN",
            document_type="credit_application",
        )
        self.assertIn("applicants.applicant2.last_name", result.fields)
        self.assertNotIn("applicants.applicant1.last_name", result.fields)

    def test_field_output_co_applicant_section_b_block(self) -> None:
        """Dealertrack-style section B must not tag applicant1 or signatures."""
        text = (
            "B. CO-APPLICANT INFORMATION\n"
            "Last Name First Name Middle Initial Social Security Number Birth Date\n"
            "HINTON SHIRRICKA 242-39-2813 01/03/1983\n"
            "Address City State Zip\n"
            "706 BERN ST NEW BERN NC 28560\n"
            "Home Phone Cell Phone Time at Address\n"
            "(252)269-2911 2 Yrs. 0 Mos. Rent/Mtg. Pmt. $0.00"
        )
        result = classify_section(text, document_type="credit_application")
        self.assertIn("applicants.applicant2.last_name", result.fields)
        self.assertIn("applicants.applicant2.ssn", result.fields)
        self.assertNotIn("applicants.applicant1.last_name", result.fields)
        self.assertNotIn("applicants.applicant1.ssn", result.fields)
        self.assertFalse(any(f.startswith("signatures.") for f in result.fields))
        self.assertNotIn("applicants.applicant2.income.amount", result.fields)

    def test_field_output_business_block(self) -> None:
        result = classify_section(
            "Trade Name of Business Tax ID\nSpringdale Developers LLC 33-3229251",
            document_type="credit_application",
        )
        self.assertIn("applicants.applicant1.business_name", result.fields)
        self.assertIn("applicants.applicant1.tax_id", result.fields)

    def test_field_output_vin(self) -> None:
        result = classify_section(
            "Year Make Vehicle Identification Number\n2020 TOYOTA 1HGBH41JXMN109186",
            document_type="credit_application",
        )
        # credit_application may not list vin at top level; content type still vehicle.
        self.assertIn("vehicle_description", result.content_types)


class TestFieldClassifier(unittest.TestCase):
    def test_load_document_field_paths_credit_app(self) -> None:
        from section_content_taxonomy import load_document_field_paths  # noqa: PLC0415

        paths = load_document_field_paths("credit_application")
        self.assertIn("applicants.applicant1.last_name", paths)
        self.assertIn("applicants.applicant1.address.street_address", paths)

    def test_classify_section_fields_direct(self) -> None:
        from section_field_classifier import _classify_section_fields_heuristic  # noqa: PLC0415

        fields = _classify_section_fields_heuristic(
            "Employer Occupation Length of Employment\nACME CORP ENGINEER 24 months",
            "credit_application",
        )
        self.assertIn("applicants.applicant1.employer_name", fields)
        self.assertIn("applicants.applicant1.occupation", fields)

    def test_has_extractable_values_label_only_rows(self) -> None:
        from section_field_classifier import (  # noqa: PLC0415
            classify_section_fields,
            has_extractable_values,
        )

        text = "Last Name\nSSN\nDate of Birth"
        self.assertFalse(has_extractable_values(text))
        self.assertEqual(classify_section_fields(text, "credit_application"), [])

    def test_has_extractable_values_legal_notices_no_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        text = (
            "FEDERAL NOTICES\n"
            "IMPORTANT INFORMATION ABOUT PROCEDURES FOR OPENING A NEW ACCOUNT "
            "If applicable to your credit transaction, Federal law requires financial "
            "institutions to obtain, verify, and record information that identifies "
            "each person who opens an account."
        )
        fields = classify_section_fields(text, "credit_application")
        self.assertEqual(fields, [])

    def test_dealertrack_6bf89f71_p13_legal_sections_no_fields(self) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        root = Path(__file__).resolve().parents[5]
        sections_json = (
            root
            / "wa577_gallery/credit_app_sections/dealertrack/6bf89f71/"
            "6bf89f71_p13_sections/6bf89f71_p13_sections.json"
        )
        payload = json.loads(sections_json.read_text())
        for section in payload["sections"]:
            fields = classify_section_fields(
                section["text"],
                "credit_application",
                content_types=section.get("content_classification", {}).get(
                    "content_types"
                ),
            )
            self.assertEqual(
                fields,
                [],
                msg=f"section {section['index']} should have no extractable fields",
            )

    def test_label_only_previous_address_empty_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        fields = classify_section_fields(
            "Yrs. Mos.\nPrevious Full Address (if less than 2 years)City State Zip",
            "credit_application",
        )
        self.assertEqual(fields, [])

    def test_legal_notices_empty_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        federal = (
            "FEDERAL NOTICES\nIMPORTANT INFORMATION ABOUT PROCEDURES FOR OPENING A NEW ACCOUNT "
            "You will be asked for your name, address, date of birth, and other information."
        )
        state = (
            "STATE NOTICES\nCalifornia Residents : An applicant, if married, may apply for a separate account.\n"
            "Married Wisconsin Residents : complete Section A about yourself and Section B about your spouse."
        )
        footer = "©2026 Dealertrack Inc. All rights reserved DT 1/26"
        self.assertEqual(classify_section_fields(federal, "credit_application"), [])
        self.assertEqual(classify_section_fields(state, "credit_application"), [])
        self.assertEqual(classify_section_fields(footer, "credit_application"), [])

    def test_legal_section_b_reference_not_co_applicant(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        fields = classify_section_fields(
            "Married Wisconsin Residents : complete Section A and Section B about your spouse.",
            "credit_application",
        )
        self.assertEqual(fields, [])

    def test_consent_prose_empty_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        text = (
            'The words "we," "us," "our" and "ours" as used below refer to us, the dealer, '
            "and to the financial institution(s) selected to receive your application. "
            "In accordance with the Fair Credit Reporting Act, you authorize that such "
            "financial institutions may submit your applications to other financial institutions."
        )
        self.assertEqual(classify_section_fields(text, "credit_application"), [])

    def test_fasttext_field_model_when_present(self) -> None:
        from section_field_classifier import (  # noqa: PLC0415
            DEFAULT_FIELD_MODEL_PATH,
            classify_section_fields,
            load_default_field_model,
        )

        if not DEFAULT_FIELD_MODEL_PATH.exists():
            self.skipTest("section_fields.bin not trained yet")
        model = load_default_field_model()
        self.assertIsNotNone(model)
        fields = classify_section_fields(
            "Last Name First Name SSN\nSMITH JOHN 123-45-6789",
            "credit_application",
            field_model=model,
        )
        self.assertIn("applicants.applicant1.last_name", fields)
        self.assertIn("applicants.applicant1.ssn", fields)

    def test_legal_consent_agreement_prose_no_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

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
        self.assertEqual(classify_section_fields(text, "credit_application"), [])

    def test_gallery_6f99d76d_legal_agreement_section_no_fields(self) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        root = Path(__file__).resolve().parents[5]
        sections_json = (
            root
            / "wa577_gallery/credit_app_sections/dealertrack/6f99d76d/"
            "6f99d76d_p2_sections/6f99d76d_p2_sections.json"
        )
        payload = json.loads(sections_json.read_text())
        legal = next(s for s in payload["sections"] if "Fair Credit Reporting" in s["text"])
        self.assertEqual(classify_section_fields(legal["text"], "credit_application"), [])

    def test_gallery_005f9437_agreement_with_embedded_phone_no_fields(self) -> None:
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        root = Path(__file__).resolve().parents[5]
        sections_json = (
            root
            / "wa577_gallery/credit_app_sections/dealertrack/005f9437/"
            "005f9437_p9_sections/005f9437_p9_sections.json"
        )
        payload = json.loads(sections_json.read_text())
        section = next(s for s in payload["sections"] if s.get("index") == 4)
        self.assertEqual(classify_section_fields(section["text"], "credit_application"), [])

    def test_form_section_still_extracts_fields(self) -> None:
        from section_field_classifier import classify_section_fields  # noqa: PLC0415

        text = (
            "Last Name First Name Social Security Number\n"
            "SMITH JOHN 123-45-6789"
        )
        fields = classify_section_fields(text, "credit_application")
        self.assertIn("applicants.applicant1.last_name", fields)
        self.assertIn("applicants.applicant1.ssn", fields)


if __name__ == "__main__":
    unittest.main()
