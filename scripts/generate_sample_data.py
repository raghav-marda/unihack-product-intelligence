"""
Generates 3 synthetic-but-realistic industrial product datasheets as PDFs.
These simulate real manufacturer spec sheets (bearing, motor, proximity sensor)
with deliberately messy/incomplete data to stress-test the extraction +
validation pipeline (missing fields, inconsistent units, unlabeled compliance).
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import os

from pathlib import Path

OUT_DIR = str(Path(__file__).resolve().parents[1] / "data" / "samples")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, spaceAfter=4)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceBefore=10, spaceAfter=4,
                     textColor=colors.HexColor('#1a3c6e'))
normal = styles['Normal']
small = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8, textColor=colors.grey)


def build_table(rows, col_widths=(65*mm, 95*mm)):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#999999')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef2f7')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def doc_header(elements, product_name, model, manufacturer, doc_no, rev, date):
    elements.append(Paragraph(manufacturer, title_style))
    elements.append(Paragraph(f"Technical Data Sheet — {product_name}", h2))
    meta = build_table([
        ["Model / Part No.", model],
        ["Document No.", doc_no],
        ["Revision", rev],
        ["Date", date],
    ])
    elements.append(meta)
    elements.append(Spacer(1, 10))


# ---------------------------------------------------------------------------
# PRODUCT 1: Deep Groove Ball Bearing — fairly complete, but mixed units
# ---------------------------------------------------------------------------
def build_bearing_pdf():
    path = os.path.join(OUT_DIR, "product_bearing_6205.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    el = []
    doc_header(el, "Deep Groove Ball Bearing", "SKB-6205-2RS", "Nordvik Bearing Industries GmbH",
               "NBI-DS-6205-2RS-R3", "Rev. 3", "2026-02-11")

    el.append(Paragraph("1. Description", h2))
    el.append(Paragraph(
        "Single row deep groove ball bearing with double-side rubber seals (2RS), "
        "pre-lubricated with high-performance grease. Suitable for general industrial "
        "machinery, electric motors, pumps, and conveyor rollers requiring moderate "
        "radial and axial load capacity at standard operating speeds.", normal))

    el.append(Paragraph("2. Dimensional Specifications", h2))
    el.append(build_table([
        ["Parameter", "Value"],
        ["Bore Diameter (d)", "25 mm"],
        ["Outside Diameter (D)", "52 mm"],
        ["Width (B)", "15 mm"],
        ["Chamfer (r min.)", "1.1 mm"],
        ["Weight", "0.130 kg"],
    ]))

    el.append(Paragraph("3. Performance Ratings", h2))
    el.append(build_table([
        ["Parameter", "Value"],
        ["Dynamic Load Rating (Cr)", "14000 N"],
        ["Static Load Rating (C0r)", "6950 N"],
        ["Limiting Speed (grease)", "18,000 rpm"],
        # deliberately inconsistent unit style vs rpm above, to test validation
        ["Reference Speed", "20000 min-1"],
        ["Operating Temperature Range", "-30 degC to 120 C"],
    ]))

    el.append(Paragraph("4. Material & Construction", h2))
    el.append(build_table([
        ["Ring & Ball Material", "Chrome Steel (SAE 52100)"],
        ["Cage Material", "Pressed Steel"],
        ["Seal Type", "2RS (Rubber, Contact Type)"],
        ["Lubricant", "Lithium-based grease, NLGI 2"],
    ]))

    el.append(Paragraph("5. Compliance & Standards", h2))
    el.append(Paragraph(
        "Manufactured in accordance with ISO 15:2017 boundary dimensions. "
        "RoHS 2011/65/EU compliant. Material certification available on request. "
        "This product is not rated for use in explosive atmospheres (ATEX).", normal))

    el.append(Paragraph("6. Packaging", h2))
    el.append(Paragraph("Standard pack: 1 unit per box, 50 boxes per carton. "
                         "Export packaging (VCI wrapped) available on request.", normal))

    el.append(Spacer(1, 14))
    el.append(Paragraph(
        "Note: Specifications subject to change without notice. Contact technical "
        "sales for application-specific load calculations.", small))

    doc.build(el)
    return path


# ---------------------------------------------------------------------------
# PRODUCT 2: Industrial 3-Phase Induction Motor — some fields MISSING on purpose
# ---------------------------------------------------------------------------
def build_motor_pdf():
    path = os.path.join(OUT_DIR, "product_motor_im100l.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    el = []
    doc_header(el, "Three-Phase Squirrel Cage Induction Motor", "IM-100L-4P",
               "Veltrix Motors & Drives Pvt. Ltd.", "VMD/TDS/IM100L/2026", "Rev. 1", "2026-05-02")

    el.append(Paragraph("1. General Description", h2))
    el.append(Paragraph(
        "TEFC (Totally Enclosed Fan Cooled) three-phase induction motor designed for "
        "continuous duty (S1) industrial applications including pumps, fans, and "
        "conveyor drives. Foot-mounted, aluminium frame construction.", normal))

    el.append(Paragraph("2. Electrical Ratings", h2))
    el.append(build_table([
        ["Parameter", "Value"],
        ["Rated Output", "3 kW"],
        ["Rated Voltage", "415 V (+/-10%)"],
        ["Frequency", "50 Hz"],
        ["Rated Current", "6.2 A"],
        # Poles omitted from table on purpose, only implied by model code "4P"
        ["Power Factor (cos phi)", "0.82"],
        ["Efficiency Class", ""],   # BLANK in source - should be flagged missing
        ["Insulation Class", "F"],
        ["Protection Rating", "IP55"],
    ]))

    el.append(Paragraph("3. Mechanical Specifications", h2))
    el.append(build_table([
        ["Frame Size", "100L"],
        ["Mounting", "B3 (Foot Mounted)"],
        ["Weight", "18.5 kg"],
        ["Shaft Diameter", "28 mm"],
        ["Noise Level", "N/A"],   # not measured / missing
    ]))

    el.append(Paragraph("4. Operating Conditions", h2))
    el.append(Paragraph(
        "Ambient temperature: -15C to +40C. Altitude: up to 1000m above sea level "
        "without derating. Duty type: S1 (Continuous).", normal))

    el.append(Paragraph("5. Compliance", h2))
    el.append(Paragraph(
        "Conforms to IEC 60034 series. CE marked. Efficiency tested per IEC 60034-2-1 "
        "(class rating pending certification update — see engineering bulletin EB-2026-07).",
        normal))

    doc.build(el)
    return path


# ---------------------------------------------------------------------------
# PRODUCT 3: Inductive Proximity Sensor — compact, electronic component style
# ---------------------------------------------------------------------------
def build_sensor_pdf():
    path = os.path.join(OUT_DIR, "product_sensor_ips18.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    el = []
    doc_header(el, "Inductive Proximity Sensor", "IPS-18-M-DC-NO", "Sentra Automation Systems",
               "SAS-DS-IPS18-2026-v2", "v2.0", "2026-06-18")

    el.append(Paragraph("1. Overview", h2))
    el.append(Paragraph(
        "M18 cylindrical inductive proximity sensor for non-contact detection of "
        "metallic objects. DC 3-wire, NPN/NO output. Designed for factory automation, "
        "packaging machinery, and material handling applications.", normal))

    el.append(Paragraph("2. Electrical Characteristics", h2))
    el.append(build_table([
        ["Parameter", "Value"],
        ["Operating Voltage", "10-30 VDC"],
        ["Sensing Range", "8 mm (Sn, non-flush)"],
        ["Output Type", "NPN, Normally Open (NO)"],
        ["Output Current (max)", "200 mA"],
        ["Switching Frequency", "500 Hz"],
        ["Response Time", "1 ms"],
        ["Reverse Polarity Protection", "Yes"],
        ["Short Circuit Protection", "Yes"],
    ]))

    el.append(Paragraph("3. Mechanical & Environmental", h2))
    el.append(build_table([
        ["Housing Material", "Nickel-plated Brass"],
        ["Thread Size", "M18 x 1"],
        ["Connection Type", "Cable, 2m PVC (3-wire)"],
        ["Protection Rating", "IP67"],
        ["Operating Temperature", "-25C to +70C"],
        ["Vibration Resistance", "55 Hz, amplitude 1mm"],
    ]))

    el.append(Paragraph("4. Compliance", h2))
    el.append(Paragraph(
        "CE, UKCA. RoHS compliant. Not certified for intrinsically safe / hazardous "
        "location use.", normal))

    el.append(Paragraph("5. Ordering Information", h2))
    el.append(build_table([
        ["Part Number", "IPS-18-M-DC-NO"],
        ["Variant (Flush Mount)", "IPS-18-M-DC-NO-F (Sn 5mm)"],
        ["Variant (NC Output)", "IPS-18-M-DC-NC"],
    ]))

    doc.build(el)
    return path


if __name__ == "__main__":
    p1 = build_bearing_pdf()
    p2 = build_motor_pdf()
    p3 = build_sensor_pdf()
    print("Generated:")
    for p in (p1, p2, p3):
        print(" -", p)
