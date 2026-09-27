#!/usr/bin/env python3
"""Read-only CLI help/portability inspection; write only its verification receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.lab import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("reports/verification/cli_help.json"))
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    results=[];started=time.monotonic()
    for script in sorted((root/"scripts").glob("*.py")):
        if script.name.startswith("_") or script.name==Path(__file__).name:continue
        check=subprocess.run([sys.executable,str(script),"--help"],cwd=root,text=True,capture_output=True,timeout=45)
        results.append(dict(script=str(script.relative_to(root)),returncode=check.returncode,help_present="usage:" in check.stdout,error=check.stderr[-2000:] if check.returncode else ""))
    broken=[r for r in results if r["returncode"] or not r["help_present"]]
    result=dict(status="passed" if not broken else "failed",cli_checks=results,seconds=time.monotonic()-started,
                scope="Argparse --help/import checks; not evidence of NIH training completion")
    atomic_json(a.output,result);print(json.dumps(result,indent=2))
    return 1 if broken else 0


if __name__=="__main__":sys.exit(main())
