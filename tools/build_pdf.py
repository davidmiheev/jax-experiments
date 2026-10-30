#!/usr/bin/env python3
"""Compile writeups to PDF: markdown -> LaTeX (pandoc) -> PDF (XeLaTeX).

Ported from ml-ladder's ``tools/build_pdf.py`` (markdown preparation and file
discovery). The look is the atcoder_lectures notes' (``tools/tex/writeupstyle.sty``:
DejaVu fonts, the blue title band, sans-serif headings, tinted boxes for the
status block and code). Every ``docs/m<N>-*.md`` writeup becomes
``docs/pdf/<name>.pdf``.

    uv run python tools/build_pdf.py --all                 # every writeup
    uv run python tools/build_pdf.py docs/m0-r1-parity.md  # one
    uv run python tools/build_pdf.py --all --keep          # keep the .tex

Requires pandoc and XeLaTeX on PATH (Debian/Ubuntu: pandoc texlive-xetex
texlive-latex-extra texlive-fonts-recommended fonts-texgyre fonts-dejavu; the OpenType TeX Gyre
fonts, not the Type 1 `tex-gyre`, for the math font). The committed PDFs
mean readers never need either.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PDF_DIR = DOCS / "pdf"

# The house style (tools/tex/writeupstyle.sty, adapted from the atcoder_lectures
# notes: fonts, palette, title band, headings, boxes, footer), then figure sizing
# and float placement, which pandoc's default template does not set.
PREAMBLE = r"""
\newcommand{\writeupseries}{jax-experiments writeups}
\newcommand{\writeupmilestone}{@MILESTONE@}
\newcommand{\writeupsource}{\detokenize{@SOURCE@}}
\usepackage{writeupstyle}
\graphicspath{{@GRAPHICS@/}}
% Keep each section's figures inside it instead of drifting forward.
\usepackage[section]{placeins}
% Full width: this repo's figures are two-panel and wide.
\setkeys{Gin}{width=\linewidth,keepaspectratio}
\captionsetup{skip=4pt}
\floatplacement{figure}{htbp}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.6}
% Let several figures share a page rather than each claiming one.
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
% The body follows the table of contents on the same page, under the band.
\let\jaxexpoldtoc\tableofcontents
\renewcommand{\tableofcontents}{\jaxexpoldtoc\vspace{10pt}\hrule\vspace{6pt}}
"""

# Fonts, geometry and colours come from writeupstyle.sty; pandoc's own \hypersetup
# runs after it, so the link colours are passed here too (the palette is defined
# by then).
PANDOC_VARIABLES = {
    "documentclass": "article",
    "fontsize": "11pt",
    "colorlinks": "true",
    "linkcolor": "writeupaccent",
    "urlcolor": "writeupaccent",
    "citecolor": "writeupaccent",
    "filecolor": "writeupaccent",
    "toccolor": "writeupink",
}
STYLE_DIR = Path(__file__).resolve().parent / "tex"


def require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(f"{tool} is not installed — see the module docstring for the packages needed")
    return path


def prepare_markdown(source: Path) -> tuple[str, str]:
    """Return the document title and a LaTeX-friendly copy of the markdown.

    Three adjustments, none of which belongs in the committed markdown: the H1
    becomes document metadata rather than a section, runs of image lines are
    split into separate paragraphs so each becomes its own float (GitHub
    renders them inline, which is what we want *there*), and the status
    blockquote gets hard line breaks so its fields do not reflow into a
    paragraph.
    """
    lines = source.read_text().splitlines()
    title = lines[0].lstrip("# ").strip()

    body: list[str] = []
    previous_was_image = False
    remaining = lines[1:]
    for index, line in enumerate(remaining):
        is_image = line.startswith("![")
        if is_image and previous_was_image:
            body.append("")
        # GitHub starts a list directly under a paragraph line; pandoc needs a
        # blank line first, or the items are run into the paragraph.
        previous = body[-1] if body else ""
        if re.match(r"(- |\d+\. )", line) and previous.strip() and not re.match(r"(\s|- |\d+\. |\||>)", previous):
            body.append("")
        # The status block is one line per field; markdown would reflow it into
        # a paragraph, so make each line a hard break.
        if line.startswith(">") and index + 1 < len(remaining) and remaining[index + 1].startswith(">"):
            line = line.rstrip() + " \\"
        body.append(line)
        previous_was_image = is_image
    return title, "\n".join(body)


def page_count(pdf: Path) -> str:
    """Page count via pdfinfo when poppler is around; silent when it is not."""
    if shutil.which("pdfinfo") is None:
        return ""
    output = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    match = re.search(r"^Pages:\s+(\d+)", output, re.M)
    return f", {match.group(1)} pages" if match else ""


def milestone(source: Path) -> str:
    """``m0-r1-parity.md`` -> ``M0``; the milestone table lives in PLAN.md."""
    match = re.match(r"(m\d+)-", source.name)
    return match.group(1).upper() if match else source.stem


def build(source: Path, *, keep_tex: bool) -> Path:
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"no writeup at {source}")

    title, markdown = prepare_markdown(source)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / f"{source.stem}.pdf"

    with tempfile.TemporaryDirectory(prefix="jaxexp-pdf-") as scratch:
        work = Path(scratch)
        (work / "body.md").write_text(markdown)
        # xelatex runs in the scratch directory, so figures need an absolute root.
        preamble = (PREAMBLE.replace("@GRAPHICS@", source.parent.as_posix())
                    .replace("@MILESTONE@", milestone(source))
                    .replace("@SOURCE@", source.relative_to(ROOT).as_posix()))
        (work / "preamble.tex").write_text(preamble)

        command = [
            require("pandoc"),
            str(work / "body.md"),
            "-o",
            str(work / "writeup.tex"),
            "--standalone",
            "--from=markdown+tex_math_dollars+pipe_tables+raw_tex",
            "--to=latex",
            "--toc",
            "--toc-depth=2",
            "--shift-heading-level-by=-1",  # "## A1" becomes a \section
            f"--resource-path={source.parent}",
            "--include-in-header",
            str(work / "preamble.tex"),
            "-M",
            f"title={title}",
            "-M",
            f"date={date.today().isoformat()}",
        ]
        for key, value in PANDOC_VARIABLES.items():
            command += ["-V", f"{key}={value}"]
        subprocess.run(command, check=True)

        # Two passes so the table of contents and float numbering settle.
        for _ in range(2):
            result = subprocess.run(
                [require("xelatex"), "-interaction=nonstopmode", "-halt-on-error", "writeup.tex"],
                cwd=work,
                env={**os.environ, "TEXINPUTS": f"{STYLE_DIR}{os.pathsep}"},  # trailing separator: keep the defaults
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-40:])
                raise SystemExit(f"xelatex failed for {source.name}:\n{tail}")

        log = (work / "writeup.log").read_text(errors="replace")
        missing = sorted(set(re.findall(r"Missing character: There is no (.+?) \(", log)))
        if missing:
            # XeTeX drops unmapped glyphs silently; never let that reach a PDF.
            raise SystemExit(f"font is missing glyphs {missing} — fix the source or change mainfont")

        shutil.copyfile(work / "writeup.pdf", destination)
        if keep_tex:
            shutil.copyfile(work / "writeup.tex", destination.with_suffix(".tex"))

    print(f"{destination.relative_to(ROOT)}  ({destination.stat().st_size / 1024:.0f} KB{page_count(destination)})")
    return destination


def writeups() -> list[Path]:
    """Every milestone writeup: ``docs/m<N>-*.md``."""
    return sorted(DOCS.glob("m[0-9]*-*.md"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", type=Path, help="writeup markdown files")
    parser.add_argument("--all", action="store_true", help="build every docs/m<N>-*.md writeup")
    parser.add_argument("--keep", action="store_true", help="also write the intermediate .tex next to the PDF")
    args = parser.parse_args(argv)
    if args.all:
        selected = writeups()
    elif args.sources:
        selected = args.sources
    else:
        parser.error("name at least one writeup, or pass --all")
    for source in selected:
        build(source, keep_tex=args.keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
