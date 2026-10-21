#!/usr/bin/env python
"""Copy the solver modules from the research repository into src/sgl_analytic.

  python scripts/sync_from_analytic.py                 # port, then write SOURCE.json
  python scripts/sync_from_analytic.py --check         # only report modules that differ
  python scripts/sync_from_analytic.py --src /path/to/stochastic_wl_analytic/analytic

The physics files are the research repository's analytic/<module>.py with ONLY the local
imports rewritten to relative package imports (import sgl -> from . import sgl, and the
sys.path insert removed). pipeline.py, api.py and cli.py are this package's own wrappers
and are not touched: carry new run() defaults over by hand. After a sync, regenerate the
references with the research repository's run_full_chain.run and run
tests/test_reference.py (the port must reproduce them bit for bit).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "src" / "sgl_analytic"
MODULES = ["sgl", "sgl_full", "subhalos", "subhalos_exact", "filaments",
           "pos_closure", "pos_closure_fused", "sgl_direct"]
LOCAL = "|".join(MODULES)
PATH_HACK = re.compile(r"^HERE = Path\(__file__\)\.resolve\(\)\.parent\n"
                       r"sys\.path\.insert\(0, str\(HERE\)\)\n", re.M)
IMPORT = re.compile(rf"^(\s*)import ({LOCAL})( as \w+)?[ \t]*$", re.M)
FROM = re.compile(rf"^(\s*)from ({LOCAL}) import ", re.M)


def port(text):
    """Source text with its local imports made relative; nothing else changes."""
    text = PATH_HACK.sub("", text)
    text = IMPORT.sub(r"\1from . import \2\3", text)
    text = FROM.sub(r"\1from .\2 import ", text)
    left = [line for line in text.splitlines()
            if re.match(rf"\s*(import|from) ({LOCAL})\b", line) or "sys.path" in line]
    if left:
        raise ValueError(f"unported import lines: {left}")
    return text


def git(src, *args):
    out = subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True)
    return out.stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=ROOT.parent / "stochastic_wl_analytic" / "analytic")
    ap.add_argument("--check", action="store_true", help="report differences, write nothing")
    a = ap.parse_args(argv)

    files, stale = {}, []
    for m in MODULES:
        raw = (a.src / f"{m}.py").read_text()
        new = port(raw)
        dest = DST / f"{m}.py"
        if not dest.is_file() or dest.read_text() != new:
            stale.append(m)
        files[f"{m}.py"] = hashlib.sha256(raw.encode()).hexdigest()
        if not a.check:
            dest.write_text(new)
    if a.check:
        print("in sync" if not stale else f"differ from {a.src}: {', '.join(stale)}")
        return 1 if stale else 0
    repo = a.src.parent
    dirty = git(repo, "status", "--porcelain", "--", "analytic/")
    record = dict(source_repository=str(repo), source_commit=git(repo, "rev-parse", "HEAD"),
                  source_branch=git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                  source_analytic_dirty=bool(dirty), synced=dt.date.today().isoformat(),
                  sha256=files)
    (ROOT / "SOURCE.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"ported {len(MODULES)} modules ({', '.join(stale) or 'none'} changed); "
          f"wrote SOURCE.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
