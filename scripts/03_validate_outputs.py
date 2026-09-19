from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(message)


def check_manifest() -> None:
    manifest_path = ROOT / "config" / "figure_manifest.tsv"
    if not manifest_path.exists():
        fail("Missing config/figure_manifest.tsv")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        fail("Figure manifest is empty")
    for row in rows:
        for key in ["output_pdf", "output_svg", "output_png"]:
            path = ROOT / row[key]
            if not path.exists() or path.stat().st_size == 0:
                fail(f"Missing output: {row[key]}")
        png_path = ROOT / row["output_png"]
        with Image.open(png_path) as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(dpi) < 590:
                fail(f"PNG dpi is below 600: {row['output_png']} {dpi}")


def main() -> None:
    check_manifest()
    print("Validation passed.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
