import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_spec_is_current():
    r = subprocess.run([sys.executable, "scripts/export_openapi.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, "run `python scripts/export_openapi.py` and commit: " + r.stdout


def test_spec_documents_endpoints_auth_and_errors():
    spec = json.loads((ROOT / "openapi.json").read_text())
    assert {"/health", "/predict", "/predict/batch", "/drift", "/model-info"} <= set(spec["paths"])
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
    post = spec["paths"]["/predict"]["post"]
    assert {"200", "401", "422"} <= set(post["responses"]) and post["security"]
    assert "security" not in spec["paths"]["/health"]["get"]
