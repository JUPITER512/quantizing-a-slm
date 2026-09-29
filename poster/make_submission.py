"""Create the STUD.IP submission: {student_id}.zip with exactly two PDFs (poster + appendix).

Rebuilds both PDFs first (poster.pptx -> poster.pdf with PowerPoint; appendix with Edge) and refuses to zip a draft (placeholder text, missing repo link or
missing signed declaration) unless --force is given.

    .venv\\Scripts\\python.exe poster\\make_submission.py --student-id 1234567
    .venv\\Scripts\\python.exe poster\\make_submission.py --student-id 1234567 --partner-id 7654321
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile

from build_poster import POSTER

def run(script: str) -> str:
    out = subprocess.run([sys.executable, str(POSTER / script)], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=POSTER)
    print(out.stdout.strip())
    return out.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--partner-id", help="second student ID for a two-person group")
    ap.add_argument("--force", action="store_true", help="zip even if a check fails (not recommended)")
    args = ap.parse_args(argv)
    for sid in filter(None, [args.student_id, args.partner_id]):
        if not re.fullmatch(r"\d+", sid):
            raise SystemExit(f"student ID must be digits only: {sid!r}")

    print("== poster"); poster_log = run("build_pptx.py")
    print("\n== appendix"); appendix_log = run("build_appendix.py")
    problems = []
    if "STATUS: DRAFT" in poster_log:
        problems.append("poster is a draft (placeholder text or repo link missing)")
    if "OVERFLOW" in poster_log:
        problems.append("a poster box overflows")
    if "pages: 1 (must be 1)" not in poster_log:
        problems.append("poster is not exactly one page")
    if "TOO SMALL" in poster_log:
        problems.append("poster text below 24 pt")
    if "STATUS: DRAFT" in appendix_log:
        problems.append("appendix has no signed declaration")
    if problems and not args.force:
        print("\nNOT ZIPPED:\n  - " + "\n  - ".join(problems))
        return 1

    name = f"{args.student_id}_{args.partner_id}.zip" if args.partner_id else f"{args.student_id}.zip"
    zpath = POSTER / name
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(POSTER / "poster.pdf", "poster.pdf")
        z.write(POSTER / "appendix.pdf", "appendix.pdf")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    print(f"\nwrote {zpath}\n  contents: {names} (exactly two PDFs: {'yes' if len(names) == 2 else 'NO'})")
    if problems:
        print("  WARNING (--force): " + "; ".join(problems))
    print("Upload it to the 'posters' folder of your seminar group on STUD.IP, then download it again and open both PDFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
