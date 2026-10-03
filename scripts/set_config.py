"""Change models/params.json safely after Validation.

  python scripts/set_config.py                         # show the current settings
  python scripts/set_config.py news_veto=false         # apply one or more changes
  python scripts/set_config.py vix_mode=level vix_threshold=25 vix_boost=1.4 --note "why"

Unknown keys and invalid values are refused (Params validation). The change and
the reason are appended to models/config_history.json, so the final materials can
show when and why the configuration changed. Then run the tests and one
`python -m eyris.live decide` to confirm the agent still decides.
"""
import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyris.config import Params  # noqa: E402

PARAMS = ROOT / "models" / "params.json"
HISTORY = ROOT / "models" / "config_history.json"


def parse(value, kind):
    if kind is bool:
        if value.lower() in ("true", "1", "on", "yes"):
            return True
        if value.lower() in ("false", "0", "off", "no"):
            return False
        raise ValueError(f"not a boolean: {value}")
    return kind(value)


def apply(changes, note="", params_file=PARAMS, history_file=HISTORY):
    """Validate and write ``changes`` ({key: str}); returns {key: (old, new)}."""
    doc = json.loads(Path(params_file).read_text())
    cur = doc["params"]
    types = {f.name: type(getattr(Params(), f.name)) for f in fields(Params)}
    new = dict(cur)
    for k, v in changes.items():
        if k not in types:
            raise KeyError(f"unknown setting {k!r}; known: {', '.join(sorted(types))}")
        new[k] = parse(v, types[k])
    Params(**new)                               # raises ValueError on an invalid combination
    diff = {k: (cur.get(k), new[k]) for k in changes if cur.get(k) != new[k]}
    if diff:
        doc["params"] = new
        Path(params_file).write_text(json.dumps(doc, indent=1))
        hist = json.loads(Path(history_file).read_text()) if Path(history_file).exists() else []
        hist.append({"time": datetime.now(timezone.utc).isoformat(), "changes": {k: list(v) for k, v in diff.items()},
                     "note": note})
        Path(history_file).write_text(json.dumps(hist, indent=1))
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("changes", nargs="*", help="key=value")
    ap.add_argument("--note", default="", help="why (kept in models/config_history.json)")
    a = ap.parse_args()
    if not a.changes:
        print(json.dumps(json.loads(PARAMS.read_text())["params"], indent=1))
        return 0
    try:
        diff = apply(dict(c.split("=", 1) for c in a.changes), a.note)
    except (KeyError, ValueError) as e:
        print(f"refused: {e}")
        return 1
    for k, (old, new) in diff.items():
        print(f"{k}: {old} -> {new}")
    print("no change" if not diff else "saved. Next: python -m pytest -q, then one `python -m eyris.live decide`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
