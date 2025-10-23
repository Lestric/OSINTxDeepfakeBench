
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_osint_list.py
----------------------
Erzeugt im Ordner ``OSINT/Research_Data`` eine Liste aller Videos im Unterordner
``videos`` im Format analog zu Celeb-DF-v2 (eine führende "1" und ein relativer Pfad).

Beispielzeile:
    1 videos/example.mp4

Die Datei wird bei jedem Aufruf **neu erzeugt** (überschrieben).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".m4v"}


def iter_videos(videos_dir: Path) -> List[Path]:
    """Liste aller Videodateien (rekursiv) mit zulässigen Endungen, stabil alphabetisch sortiert.
    Groß-/Kleinschreibung wird ignoriert.
    """
    files = [
        p for p in videos_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    files.sort(key=lambda p: str(p).lower())
    return files


def generate_osint_list(research_data_dir: Path, list_filename: str = "List_of_testing_videos.txt",
                        videos_subdir: str = "videos") -> Path:
    """Erzeugt die Listendatei im Research_Data-Ordner.

    Args:
        research_data_dir: Pfad zu ``.../OSINT/Research_Data``
        list_filename: Name der zu erzeugenden Textdatei.
        videos_subdir: Name des Unterordners, der die Videos enthält (Standard: "videos").
    Returns:
        Pfad zur erzeugten Textdatei.
    """
    research_data_dir = research_data_dir.resolve()
    videos_dir = (research_data_dir / videos_subdir).resolve()

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos-Ordner nicht gefunden: {videos_dir}")

    out_file = research_data_dir / list_filename
    videos = iter_videos(videos_dir)

    # Schreibe im gewünschten Format: "1 <relativer Pfad innerhalb Research_Data>\n"
    lines = []
    for v in videos:
        # relativer Pfad relativ zu Research_Data
        rel = v.relative_to(research_data_dir)
        lines.append(f"1 {rel.as_posix()}")

    out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_file


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Erzeuge OSINT-Video-Liste analog Celeb-DF-v2.")
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Pfad zum Repo-Stamm (z. B. /home/user/OSINTxDeepfakeBench-main). "
             "Darin wird OSINT/Research_Data erwartet.",
    )
    ap.add_argument(
        "--videos-subdir",
        default="videos",
        help="Name des Video-Unterordners unter OSINT/Research_Data (Standard: videos).",
    )
    ap.add_argument(
        "--filename",
        default="List_of_testing_videos.txt",
        help="Dateiname der zu erzeugenden Liste (Standard: List_of_testing_videos.txt).",
    )
    return ap.parse_args(argv)


def main(argv: Iterable[str] = None) -> int:
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    research_data_dir = Path(ns.root) / "OSINT" / "Research_Data"
    out_file = generate_osint_list(research_data_dir, list_filename=ns.filename, videos_subdir=ns.videos_subdir)
    print(str(out_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
