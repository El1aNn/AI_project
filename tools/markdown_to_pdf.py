#!/usr/bin/env python3
"""Convert the project Markdown reports to simple, readable PDFs."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def register_font() -> str:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ReportFont", str(path)))
            return "ReportFont"
        except Exception:
            continue
    return "Helvetica"


def make_styles(font_name: str):
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=font_name,
            fontSize=19,
            leading=25,
            spaceAfter=10,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=17,
            leading=22,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "Heading3Custom",
            parent=base["Heading3"],
            fontName=font_name,
            fontSize=12.2,
            leading=16,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.8,
            leading=15.2,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.8,
            leading=15,
            leftIndent=14,
            firstLineIndent=0,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.8,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.whitesmoke,
            borderColor=colors.lightgrey,
            borderWidth=0.25,
            borderPadding=4,
            spaceBefore=4,
            spaceAfter=6,
        ),
        "table": ParagraphStyle(
            "TableCellCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.3,
            leading=10.2,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.5,
            leading=10.5,
            textColor=colors.black,
        ),
    }
    return styles


INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def inline_markup(text: str) -> str:
    escaped = html.escape(text)
    return INLINE_CODE_RE.sub(
        lambda m: f'<font name="Courier" backColor="#f2f2f2">{html.escape(m.group(1))}</font>',
        escaped,
    )


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_delimiter(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def make_table(lines: Sequence[str], styles, available_width: float):
    rows = [split_table_row(line) for line in lines if line.strip().startswith("|")]
    if len(rows) >= 2 and is_table_delimiter(lines[1]):
        rows = [rows[0], *rows[2:]]
    if not rows:
        return []
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]
    data = []
    for row_idx, row in enumerate(normalized):
        style = styles["table_header"] if row_idx == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    col_width = available_width / max(1, col_count)
    table = Table(data, colWidths=[col_width] * col_count, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c0cc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [table, Spacer(1, 5)]


def flush_paragraph(buffer: list[str], story: list, styles) -> None:
    if not buffer:
        return
    text = " ".join(part.strip() for part in buffer if part.strip())
    if text:
        story.append(Paragraph(inline_markup(text), styles["body"]))
    buffer.clear()


def markdown_to_story(markdown: str, styles, available_width: float) -> list:
    story: list = []
    paragraph: list[str] = []
    lines = markdown.splitlines()
    i = 0
    first_heading = True
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            flush_paragraph(paragraph, story, styles)
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"]))
            i += 1
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and is_table_delimiter(lines[i + 1]):
            flush_paragraph(paragraph, story, styles)
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            story.extend(make_table(table_lines, styles, available_width))
            continue

        if not line.strip():
            flush_paragraph(paragraph, story, styles)
            i += 1
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading_match:
            flush_paragraph(paragraph, story, styles)
            level = len(heading_match.group(1))
            text = inline_markup(heading_match.group(2).strip())
            if level == 1 and first_heading:
                story.append(Paragraph(text, styles["title"]))
                first_heading = False
            elif level == 1:
                story.append(Paragraph(text, styles["h1"]))
            elif level == 2:
                story.append(Paragraph(text, styles["h2"]))
            else:
                story.append(Paragraph(text, styles["h3"]))
            i += 1
            continue

        bullet_match = re.match(r"^\s*-\s+(.*)$", line)
        if bullet_match:
            flush_paragraph(paragraph, story, styles)
            story.append(Paragraph(inline_markup(bullet_match.group(1)), styles["bullet"], bulletText="•"))
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph(paragraph, story, styles)
    return story


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(doc.pagesize[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def convert_file(input_path: Path, output_path: Path) -> None:
    font_name = register_font()
    styles = make_styles(font_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=input_path.stem,
    )
    markdown = input_path.read_text(encoding="utf-8")
    story = markdown_to_story(markdown, styles, doc.width)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inputs: Iterable[Path] = args.inputs or sorted(Path("reports").glob("*.md"))
    for input_path in inputs:
        output_path = args.output_dir / (input_path.stem + ".pdf")
        convert_file(input_path, output_path)


if __name__ == "__main__":
    main()
