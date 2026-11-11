"""Write the OpenAPI specification (JSON and YAML) from the live app.

    python scripts/export_openapi.py          # write openapi.json / openapi.yaml
    python scripts/export_openapi.py --check  # exit 1 if the files are stale
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import create_app          # noqa: E402
from app.settings import Settings        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def spec() -> dict:
    return create_app(Settings(api_keys=("x",))).openapi()    # no model needed


def main(argv):
    s = spec()
    js, ym = json.dumps(s, indent=2, sort_keys=True) + "\n", yaml.safe_dump(s, sort_keys=True)
    if "--check" in argv:
        ok = (ROOT / "openapi.json").read_text() == js
        print("openapi.json is up to date" if ok else "openapi.json is STALE")
        return 0 if ok else 1
    (ROOT / "openapi.json").write_text(js)
    (ROOT / "openapi.yaml").write_text(ym)
    print("wrote openapi.json and openapi.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
