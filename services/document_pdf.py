"""Generowanie estetycznych PDF z pism kadrowych."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

from agents.models import DocumentGenerateOutput
from config import BASE_DIR, CompanyProfile

FONTS_DIR = BASE_DIR / "assets" / "fonts"
FONT_REGULAR = FONTS_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONTS_DIR / "DejaVuSans-Bold.ttf"

COLOR_HEADER = (26, 47, 82)  # metal-900
COLOR_ACCENT = (47, 108, 181)  # metal-600
COLOR_TEXT = (30, 41, 59)
COLOR_MUTED = (100, 116, 139)


def _resolve_font_paths() -> tuple[Path, Path]:
    if FONT_REGULAR.exists() and FONT_BOLD.exists():
        return FONT_REGULAR, FONT_BOLD
    win_regular = Path("C:/Windows/Fonts/arial.ttf")
    win_bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if win_regular.exists() and win_bold.exists():
        return win_regular, win_bold
    raise FileNotFoundError(
        "Brak czcionek Unicode. Umieść DejaVuSans.ttf w assets/fonts/ "
        "lub uruchom na Windows z Arial."
    )


def _sanitize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", _sanitize(text))
    return [b.strip() for b in blocks if b.strip()]


class LetterPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Body", "", 8)
        self.set_text_color(*COLOR_MUTED)
        self.cell(0, 8, f"MetalTech Sp. z o.o. — strona {self.page_no()}", align="C")


def build_document_pdf(
    doc: DocumentGenerateOutput,
    company: CompanyProfile,
) -> bytes:
    regular, bold = _resolve_font_paths()
    pdf = LetterPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(22, 18, 22)
    pdf.add_page()

    pdf.add_font("Body", "", str(regular))
    pdf.add_font("Body", "B", str(bold))

    # Nagłówek firmy
    pdf.set_font("Body", "B", 14)
    pdf.set_text_color(*COLOR_HEADER)
    pdf.cell(0, 8, company.nazwa, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Body", "", 9)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(0, 5, company.adres, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"NIP: {company.nip}  |  REGON: {company.regon}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"{company.email_kadry}  |  {company.telefon_kadry}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_draw_color(*COLOR_ACCENT)
    pdf.set_line_width(0.6)
    pdf.line(22, pdf.get_y(), 188, pdf.get_y())
    pdf.ln(8)

    # Tytuł
    pdf.set_font("Body", "B", 13)
    pdf.set_text_color(*COLOR_ACCENT)
    pdf.multi_cell(0, 7, _sanitize(doc.tytul), align="C")
    pdf.ln(6)

    # Meta
    pdf.set_font("Body", "", 10)
    pdf.set_text_color(*COLOR_TEXT)
    pdf.cell(0, 6, f"Miejscowość, dnia {doc.data_pisma}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Body", "B", 10)
    pdf.cell(0, 6, f"Do: {_sanitize(doc.adresat)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Treść
    pdf.set_font("Body", "", 10)
    pdf.set_text_color(*COLOR_TEXT)
    for para in _paragraphs(doc.tresc):
        pdf.multi_cell(0, 5.5, para, align="J")
        pdf.ln(3)

    # Podstawa prawna
    if doc.podstawy_prawne:
        pdf.ln(4)
        pdf.set_font("Body", "B", 10)
        pdf.set_text_color(*COLOR_ACCENT)
        pdf.cell(0, 6, "Podstawa prawna:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Body", "", 9)
        pdf.set_text_color(*COLOR_TEXT)
        for item in doc.podstawy_prawne:
            pdf.cell(0, 5, f"• {item}", new_x="LMARGIN", new_y="NEXT")

    # Podpis
    pdf.ln(12)
    pdf.set_font("Body", "", 10)
    pdf.multi_cell(0, 5, _sanitize(doc.podpis))
    pdf.ln(8)
    pdf.set_font("Body", "", 9)
    pdf.set_text_color(*COLOR_MUTED)
    pdf.cell(0, 5, company.reprezentant, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, company.nazwa, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())
