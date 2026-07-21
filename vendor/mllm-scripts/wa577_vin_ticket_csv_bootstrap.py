#!/usr/bin/env python3
"""Bootstrap CSV files for wa577_vin_ticket_csv_crop from Linear attachment IDs.

Requires LINEAR_API_KEY or uses pre-seeded files in wa577_gallery/vin_ticket_csv_crop/csvs/.
Run via Cursor Linear MCP get_attachment when API key unavailable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "wa577_gallery/vin_ticket_csv_crop/csvs"

# attachment_id -> output filename (prefer triage, else smallest Looker sample)
ATTACHMENTS: dict[str, list[tuple[str, str]]] = {
    "WA-489": [("bf0e6d7d-66d2-418a-a961-5794b8768892", "wa489_triage.csv"),
               ("d66340d0-3ad1-4385-ba6e-797030ad5a64", "wa489_looker.csv")],
    "WA-411": [("8195de21-0f40-4b60-9554-943bdd29bb23", "wa411_triage.csv"),
               ("7347fc01-65d4-4635-a41c-836e4767fc9c", "wa411_looker.csv")],
    "WA-538": [("28bcf62f-af15-45e4-9bdd-9812928c36f5", "wa538_looker.csv")],
    "WA-477": [("27c17eff-efb5-4ccc-9896-13d4cbfd90a6", "wa477_triage.csv")],
    "WA-539": [("b3d0b2cd-cd7b-4245-8a95-1b7076576025", "wa539_triage.csv")],
    "WA-553": [("ea6bb4d5-ee27-412c-bc29-89bac139af9e", "wa553_triage.csv")],
    "WA-244": [("b64147ba-9ce8-43ba-82da-cf9d0e6ebe11", "wa244_ticket.csv")],
    "WA-231": [("3e78ee62-c05c-4419-bddf-dc5fddeaaf47", "wa231_hitl.csv")],
    "WA-221": [("73258bc1-f218-4921-8be5-eeb601dbd787", "wa221_hitl.csv")],
    "WA-494": [("b664f6e3-0d6d-4644-a60a-01073187120b", "wa494_triage.csv")],
    "WA-528": [("d1198057-bbab-4a49-b567-3b4fc0fbb9f0", "wa528_triage.csv")],
    "WA-524": [("860d160e-0c2e-4831-b5b4-33a35ded323d", "wa524_triage.csv")],
    "WA-609": [("6309be8a-8ab2-48b8-97a2-5ce4ef48abb8", "wa609_triage.csv")],
    "WA-610": [("71e28478-b4d4-40ad-8358-290f01bcbc34", "wa610_triage.csv")],
    "WA-530": [("f674f6f2-8f84-4c37-9fd0-e9bb62aec9be", "wa530_triage.csv")],
    "WA-540": [("290ba5a4-e2d6-4dc4-9ea4-a9eb302a72d4", "wa540_triage.csv")],
    "WA-439": [("7e3c439d-bd55-43ca-a2d9-86ad1e2902a1", "wa439_triage.csv")],
}


def fetch_linear_attachment(attachment_id: str) -> str:
    api_key = os.environ.get("LINEAR_API_KEY") or os.environ.get("LINEAR_API_TOKEN")
    if not api_key:
        raise RuntimeError("LINEAR_API_KEY not set — seed CSVs via MCP get_attachment")
    import urllib.request

    query = """
    query($id: String!) {
      attachment(id: $id) { url }
    }
    """
    body = json.dumps({"query": query, "variables": {"id": attachment_id}}).encode()
    req = urllib.request.Request(
        "https://api.linear.app/graphql",
        data=body,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    url = data["data"]["attachment"]["url"]
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"tickets": {}}
    for ticket, pairs in ATTACHMENTS.items():
        manifest["tickets"][ticket] = []
        for att_id, fname in pairs:
            dest = OUT / fname
            if dest.exists() and dest.stat().st_size > 50:
                manifest["tickets"][ticket].append({"file": fname, "status": "exists"})
                continue
            try:
                content = fetch_linear_attachment(att_id)
                dest.write_text(content)
                manifest["tickets"][ticket].append({"file": fname, "status": "downloaded"})
                print(f"Wrote {dest}", flush=True)
            except Exception as exc:
                manifest["tickets"][ticket].append({"file": fname, "status": f"error: {exc}"})
                print(f"SKIP {fname}: {exc}", flush=True)
    (OUT.parent / "csv_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    from typing import Any
    main()
